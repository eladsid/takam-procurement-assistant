/**
 * embed.ts — turning text into vectors, behind ONE interface. (AD-2)
 *
 * Everything in the retrieval path (ingest, ask, the Lambda) talks to this file
 * and to nothing else about embeddings. That is deliberate: an embedding is the
 * single hardest dependency to swap after the fact, because vectors from two
 * different models are not comparable — change the model without re-indexing and
 * search does not error, it just quietly returns nonsense. So the provider lives
 * behind `EmbedTransport`, the choice lives in one config value, and the dims are
 * asserted on every single call (see `assertDims`).
 *
 * Three transports are implemented:
 *   bedrock:titan   — THE DEFAULT since 30/08/2026. Real AWS, 1024 dims, one text
 *                     per API call fanned out across EMBED_CONCURRENCY.
 *   bedrock:cohere  — the previous default. Real AWS, multilingual, 96 per call.
 *   gemini          — outside AWS, REST only. A parachute, not a destination.
 *
 * Why the default moved: it was measured, not assumed. On four Hebrew query pairs
 * Titan separated the relevant instruction from an irrelevant one 3.5x better than
 * Cohere (0.393 vs 0.111 average gap), at a fifth of the price and the same dims.
 * The full numbers are on `titanTransport`.
 *
 * Why "blocked" never means "Bedrock is closed" — this cost real time to learn.
 * Bedrock quotas are PER MODEL, not per service, and two failures look alike:
 *   - "Too many requests"        → a requests-per-minute ceiling. Transient.
 *     Waiting helps. This is what backoff is for.
 *   - "Too many tokens per day"  → a DAILY token ceiling. Not transient.
 *     Waiting 64 seconds helps exactly as much as waiting 0 seconds.
 * Both arrive as `ThrottlingException`, which is why the retry loop below reads
 * the message and not just the error name.
 *
 * A THIRD failure exists and is neither: `AccessDeniedException: not available for
 * this account`. That one is a commercial account limit, and no amount of retrying,
 * pacing, or accepting the model's EULA changes it (verified 30/08/2026 — the
 * agreement call succeeds and the invoke is still refused).
 */

import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  EMBED_TRANSPORT, EMBED_REGION, EMBED_MODELS, EMBED_BATCH, EMBED_PACE_MS,
  EMBED_CONCURRENCY, EMBED_REGIONS, GEMINI_API_KEY,
} from "./config.js";
import type { EmbedResult, EmbedTransport } from "./types.js";
import { recordRetryWait, recordRejectedCall } from "./retry-meter.js";

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/**
 * A rough token count, used for two things only: the cost readout when the
 * provider does not report usage, and the "this chunk may overflow" warning.
 * It is never used to decide anything the model itself can decide.
 *
 * Why a hand-rolled estimate instead of a tokenizer library: the real tokenizer
 * for Cohere Multilingual v3 is not shipped with the SDK, and pulling a WASM
 * tokenizer into a Lambda bundle to get a number we only print is a bad trade.
 *
 * The ratios: Hebrew is written without vowels and its subword pieces land
 * around 2.5 characters per token, while Latin script averages closer to 4.
 * Splitting the count by script instead of using one blended ratio matters here,
 * because the same code embeds Hebrew TAKAM instructions AND English AWS docs —
 * a single ratio would be wrong for both corpora in opposite directions.
 */
export function estimateTokens(text: string): number {
  let hebrew = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0)!;
    // Hebrew block U+0590–U+05FF (letters, niqqud, punctuation like maqaf).
    if (code >= 0x0590 && code <= 0x05ff) hebrew++;
  }
  const other = text.length - hebrew;
  return Math.ceil(hebrew / 2.5 + other / 4);
}

// ---------------------------------------------------------------------------
// Dimension guard
// ---------------------------------------------------------------------------

/**
 * Fail loudly when a vector is not the length the config promised.
 *
 * This is the most important five lines in the file. Cosine similarity between
 * a 1024-vector and a 1536-vector does not throw in most implementations — it
 * either crashes far away from the cause or, worse, silently compares the first
 * N components and returns a plausible-looking score. An index built half with
 * one model and half with another is corrupt in a way no test output shows.
 * So the check happens on every batch, at the moment the vectors arrive, where
 * the error message can still name the transport that produced them.
 */
export function assertDims(vectors: number[][], expectedDims: number): void {
  for (let i = 0; i < vectors.length; i++) {
    const got = vectors[i]?.length ?? 0;
    if (got !== expectedDims) {
      throw new Error(
        `אי-התאמת ממדים בווקטור ${i}: המודל החזיר ${got} מספרים, וההגדרה מצפה ל-${expectedDims}. ` +
        `זה כמעט תמיד אומר שמודל ה-embedding הוחלף בלי לבנות מחדש את האינדקס. ` +
        `ערבוב שני מודלים באותו אינדקס לא מחזיר שגיאה בחיפוש — הוא פשוט מחזיר תוצאות שגויות, ` +
        `ולכן אנחנו עוצרים כאן. פתרון: החזר את EMBED_TRANSPORT לערך שאיתו נבנה האינדקס, או הרץ ingest מחדש.`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Error classification — throttle vs everything else
// ---------------------------------------------------------------------------

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms));

/**
 * A throttle raised by a non-AWS provider, normalised so the one retry loop
 * below can treat Gemini's HTTP 429 exactly like Bedrock's ThrottlingException.
 */
class ThrottleError extends Error {
  constructor(message: string) {
    super(message);
    // Same `name` the AWS SDK uses, so `classifyError` needs no provider branch.
    this.name = "ThrottlingException";
  }
}

type ErrorKind = "throttle" | "daily-ceiling" | "other";

/**
 * Sort a failure into: retry it, stop for today, or stop for good.
 *
 * Two rules, and both were learned by losing work to them:
 *
 * 1. A daily token ceiling ALSO arrives as `ThrottlingException`, so matching on
 *    the error name alone treats it as transient and burns eight increasingly
 *    long sleeps (over two minutes) before surfacing something that was never
 *    going to resolve today. So the message is read first, and it fails fast.
 *
 * 2. Everything else defaults to RETRY, and only a short list of genuinely
 *    permanent errors stops the run. The reasoning is in the function body — it
 *    is the single most important decision in this file and it is not obvious.
 */
function classifyError(err: any): ErrorKind {
  const message = String(err?.message ?? "");
  if (/tokens per day|daily.*(quota|limit)/i.test(message)) return "daily-ceiling";
  /**
   * The list below is of PERMANENT errors, and the default is "retry". That is
   * the opposite of how this function started, and the inversion was earned:
   * three separate ingests died on 30/08/2026 because an unlisted error name
   * defaulted to permanent —
   *
   *   ServiceUnavailableException  (503)  killed aws-lambda at ~4,000 chunks
   *   ThrottlingException under contention killed aws-bedrock-agentcore
   *   ModelErrorException  ("Try your request again") killed it a second time
   *
   * Each time the fix was "add one more name", and each time a different name
   * arrived. An allow-list of transient errors can only ever be as complete as
   * yesterday's incidents; the set of permanent errors, by contrast, is small,
   * stable and knowable — you are not allowed in, the input is invalid, the
   * thing does not exist.
   *
   * The asymmetry justifies it: retrying a permanent error wastes ~2 minutes of
   * backoff and then surfaces the same message. Abandoning a transient one
   * throws away an hour of paid embedding. Guessing "transient" is the cheap
   * mistake, so it is the correct default.
   */
  const permanent =
    err?.name === "AccessDeniedException" ||
    err?.name === "ValidationException" ||
    err?.name === "ResourceNotFoundException" ||
    err?.name === "UnrecognizedClientException" ||
    err?.name === "SerializationException" ||
    err?.$metadata?.httpStatusCode === 400 ||
    err?.$metadata?.httpStatusCode === 401 ||
    err?.$metadata?.httpStatusCode === 403 ||
    err?.$metadata?.httpStatusCode === 404;
  return permanent ? "other" : "throttle";
}

// ---------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------

/**
 * The public contract (`EmbedTransport`) plus one internal field: the model's
 * input window, used only for the overflow warning. It is deliberately NOT on
 * the shared interface — it is diagnostics, not part of what callers need.
 */
interface LocalTransport extends EmbedTransport {
  /** The model's context window, in tokens. Overflow is truncated, not rejected. */
  readonly maxInputTokens: number;
  /**
   * A HARD character limit the API validates before the model ever runs.
   * This is a different limit from `maxInputTokens` and it behaves differently:
   * exceed the token window and you get a silently shortened vector; exceed this
   * and you get a 400. See `cohereTransport` for the measured value.
   */
  readonly maxInputChars: number;
}

/**
 * One Bedrock client per region, created on first use. Lazy on purpose: in
 * `gemini` mode there may be no AWS credentials at all, and constructing a
 * client at import time would turn a working configuration into a startup crash.
 */
const bedrockClients = new Map<string, BedrockRuntimeClient>();
function bedrock(region: string): BedrockRuntimeClient {
  let client = bedrockClients.get(region);
  if (!client) {
    // Same socket clocks as the generation client (see answer.ts): a DNS blip
    // must fail in seconds, not hold an embedding call open indefinitely.
    client = new BedrockRuntimeClient({
      region,
      requestHandler: { connectionTimeout: 5_000, requestTimeout: 60_000 },
    });
    bedrockClients.set(region, client);
  }
  return client;
}

/** Shared plumbing for both Bedrock models: send JSON, get JSON back. */
async function invokeBedrock(region: string, modelId: string, body: unknown): Promise<any> {
  const res = await bedrock(region).send(new InvokeModelCommand({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(body),
  }));
  return JSON.parse(new TextDecoder().decode(res.body));
}

/**
 * bedrock:cohere — `cohere.embed-multilingual-v3`. The default, and the only
 * embedding model this account can actually call.
 *
 * Verified live against this account on 21/08/2026 with Hebrew text:
 *   request  { texts: string[], input_type: "search_document"|"search_query", truncate: "END" }
 *   response { id, texts, embeddings: number[][], response_type: "embeddings_floats" }
 *   → 1024 dims, 3 texts in one call, and the SAME sentence embedded as a
 *     document vs as a query returned vectors at cosine 0.859 — i.e. `input_type`
 *     genuinely changes the output and is not a decorative field.
 *
 * That is why `purpose` is threaded all the way from the caller: Cohere places
 * stored passages and search queries in deliberately different positions in the
 * space, and mismatching them costs real retrieval quality on exactly the kind of
 * short Hebrew question this POC is built to answer.
 *
 * Note the response carries NO token count. Cohere bills by input token but does
 * not report it here, so the cost readout uses `estimateTokens` for this
 * transport. The number printed after an ingest is an estimate, not an invoice.
 */
const cohereTransport: LocalTransport = {
  id: "bedrock:cohere",
  modelId: EMBED_MODELS["bedrock:cohere"].modelId,
  region: EMBED_REGION,
  dims: EMBED_MODELS["bedrock:cohere"].dims,
  maxBatch: EMBED_MODELS["bedrock:cohere"].maxBatch,
  // Cohere Multilingual v3 accepts ~512 tokens per text. CHUNK_CHARS is sized
  // against this same number; the warning below catches the case where it is not.
  maxInputTokens: 512,
  /**
   * Measured, not guessed: sending a 4,000-character Hebrew string on 21/08/2026
   * returned `ValidationException: Malformed input request: #/texts/0: expected
   * maxLength: 2048, actual: 4000`. So `truncate:"END"` protects the ~512-TOKEN
   * window but does NOT protect this 2,048-CHARACTER ceiling — the request is
   * rejected at validation, before the model sees it. Two different limits, two
   * different failure modes, and only one of them is silent.
   */
  maxInputChars: 2048,

  async embed(texts: string[], purpose: "document" | "query"): Promise<EmbedResult> {
    const parsed = await invokeBedrock(this.region, this.modelId, {
      texts,
      input_type: purpose === "document" ? "search_document" : "search_query",
      // Covers the ~512-token window only: over-window text is shortened instead
      // of failing. It does NOT cover the 2,048-character cap, which is rejected
      // at validation — `embedAll` checks that one itself, up front.
      truncate: "END",
    });
    const vectors = parsed.embeddings as number[][];
    assertDims(vectors, this.dims);
    return { vectors, tokens: texts.reduce((sum, t) => sum + estimateTokens(t), 0) };
  },
};

/**
 * bedrock:titan — `amazon.titan-embed-text-v2:0`. THE DEFAULT since 30/08/2026.
 *
 * The quota that used to block this opened, and measuring the two models against
 * each other changed the default. On four Hebrew query pairs — each a real
 * question, the instruction that answers it, and an unrelated instruction — the
 * gap between the relevant and the irrelevant score was:
 *
 *   cohere.embed-multilingual-v3   0.111 average   (was the default)
 *   cohere.embed-v4:0              0.289 average
 *   amazon.titan-embed-text-v2:0   0.393 average   ← 3.5x the separation
 *
 * Cohere scored an entirely unrelated instruction (leave policy, against a
 * procurement question) at 0.586 — against a Hebrew threshold of 0.60. That is
 * noise arriving one hundredth below the gate, and it is why the retrieved
 * chunks all clustered in a narrow 0.62-0.66 band. Titan is also 5x cheaper
 * ($0.02 vs $0.10 per 1M tokens) and lands on the SAME 1024 dims, so the index
 * envelope and the store format are unchanged.
 *
 * Same dims does NOT mean compatible: these are different vector spaces, and
 * mixing them silently returns nonsense. `assertDims` cannot catch that, which is
 * why the transport id is recorded in the index envelope and a switch requires a
 * full re-ingest.
 *
 * Titan takes ONE text per call at the API level, so the loop lives inside the
 * transport. The caller never learns the difference — that is the point of the
 * interface. Titan DOES report usage (`inputTextTokenCount`), so unlike Cohere
 * its token tally is exact rather than estimated.
 */
const titanTransport: LocalTransport = {
  id: "bedrock:titan",
  modelId: EMBED_MODELS["bedrock:titan"].modelId,
  region: EMBED_REGION,
  dims: EMBED_MODELS["bedrock:titan"].dims,
  maxBatch: EMBED_MODELS["bedrock:titan"].maxBatch,
  maxInputTokens: 8192,
  // Titan v2's documented ceiling is 8,192 tokens with no separate character cap;
  // unverifiable here while the model is quota-blocked, so it is set generously
  // rather than invented precisely.
  maxInputChars: 40_000,

  async embed(texts: string[], _purpose: "document" | "query"): Promise<EmbedResult> {
    // Titan has no document/query distinction, so `purpose` is discarded here —
    // deliberately, and visibly, rather than silently ignored somewhere upstream.
    //
    // One text per call is Titan's hard limit, so the only way to make a 39,000
    // chunk re-ingest finish in minutes instead of hours is to overlap the calls.
    // The vectors are written to a FIXED INDEX rather than pushed: the transport
    // contract is that vectors[i] belongs to texts[i], and concurrent completion
    // order is arbitrary. Pushing here would silently pair every chunk with the
    // wrong embedding — a corruption that no dimension check would catch.
    const vectors: number[][] = new Array(texts.length);
    let tokens = 0;

    const workers = Math.max(1, Math.min(EMBED_CONCURRENCY, texts.length));
    let next = 0;
    await Promise.all(
      Array.from({ length: workers }, async () => {
        for (;;) {
          const i = next++;
          if (i >= texts.length) return;

          /**
           * The backoff lives HERE, per call, and not only in `embedWithRetry`
           * around the whole batch — that arrangement was measured and it makes
           * throttling worse, not better. When one call out of six is throttled,
           * a batch-level retry re-sends all six, so five successful calls are
           * paid for twice and the pressure that caused the throttle goes UP.
           * Retrying the single call that failed lets the other workers keep
           * draining the queue while this one waits.
           */
          for (let attempt = 0; ; attempt++) {
            /**
             * The region rotates per call AND per retry. Rotating per call is the
             * throughput win — the rate ceiling is per region, so N regions are N
             * ceilings. Rotating on RETRY matters just as much and is easy to
             * miss: being throttled means THIS region is saturated, so retrying
             * against it is the one choice guaranteed not to help. Moving to the
             * next region usually succeeds immediately instead of sleeping.
             *
             * Safe only because Titan returns byte-identical vectors from every
             * region — see EMBED_REGIONS in config.ts for that measurement.
             */
            const region = EMBED_REGIONS[(i + attempt) % EMBED_REGIONS.length] ?? this.region;
            const attemptFrom = Date.now();
            try {
              const parsed = await invokeBedrock(region, this.modelId, { inputText: texts[i] });
              vectors[i] = parsed.embedding as number[];
              tokens += Number(parsed.inputTextTokenCount ?? 0);
              break;
            } catch (err: any) {
              // Only capacity errors are worth retrying; a daily ceiling or an
              // access error is re-thrown for embedWithRetry to classify and
              // explain, exactly as before.
              if (classifyError(err) !== "throttle" || attempt >= 7) throw err;
              recordRejectedCall("embed", Date.now() - attemptFrom, (err as Error)?.name ?? "unknown");
              // Short sleep: the next attempt hits a DIFFERENT region, so this is
              // a courtesy pause, not a wait for a ceiling to reset.
              const waitedFrom = Date.now();
              await sleep(2 ** attempt * 100);
              // Charged to the retry meter when this runs inside ask() (the query
              // embedding); during ingest there is no ledger and this is a no-op.
              recordRetryWait("embed", Date.now() - waitedFrom, (err as Error)?.name ?? "unknown");
            }
          }
        }
      }),
    );

    assertDims(vectors, this.dims);
    return { vectors, tokens };
  },
};

/**
 * gemini — `gemini-embedding-001` over plain REST.
 *
 * No SDK on purpose: one `fetch` against a documented endpoint is less to
 * install, less to bundle into a Lambda, and less to explain to a reviewer
 * than another vendor client. Node 24 has `fetch` built in.
 *
 * This is the secondary/fallback route. It is outside AWS, which matters for a
 * government demo, so it exists to keep the POC demonstrable if Bedrock ever
 * closes entirely — not as a preferred path. Its 1536 dims also mean switching to
 * it REQUIRES a full re-ingest; `assertDims` will refuse to mix the two.
 */
const geminiTransport: LocalTransport = {
  id: "gemini",
  modelId: EMBED_MODELS.gemini.modelId,
  // Gemini's endpoint is not regional in the AWS sense. "global" is recorded in
  // the index envelope instead of pretending an AWS region produced these.
  region: "global",
  dims: EMBED_MODELS.gemini.dims,
  maxBatch: EMBED_MODELS.gemini.maxBatch,
  maxInputTokens: 2048,
  maxInputChars: 20_000,

  async embed(texts: string[], _purpose: "document" | "query"): Promise<EmbedResult> {
    if (!GEMINI_API_KEY) {
      throw new Error(
        "EMBED_TRANSPORT=gemini אבל GEMINI_API_KEY ריק. " +
        "הוסף את המפתח ל-poc/.env (או חזור ל-EMBED_TRANSPORT=bedrock:cohere שרץ על AWS).",
      );
    }

    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${this.modelId}:batchEmbedContents`,
      {
        method: "POST",
        headers: { "x-goog-api-key": GEMINI_API_KEY, "content-type": "application/json" },
        body: JSON.stringify({
          requests: texts.map(text => ({
            model: `models/${this.modelId}`,
            content: { parts: [{ text }] },
          })),
        }),
      },
    );

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      const message = `Gemini embed נכשל (HTTP ${res.status}): ${detail.slice(0, 400)}`;
      // 429 is a capacity signal like any other; normalising it here lets the one
      // retry loop below handle Google and AWS with the same code.
      if (res.status === 429) throw new ThrottleError(message);
      throw new Error(message);
    }

    const parsed = await res.json() as { embeddings?: { values: number[] }[] };
    const vectors = (parsed.embeddings ?? []).map(e => e.values);
    assertDims(vectors, this.dims);
    // Gemini does not report usage on this endpoint either — same estimate as Cohere.
    return { vectors, tokens: texts.reduce((sum, t) => sum + estimateTokens(t), 0) };
  },
};

const TRANSPORTS: Record<string, LocalTransport> = {
  "bedrock:cohere": cohereTransport,
  "bedrock:titan": titanTransport,
  gemini: geminiTransport,
};

/**
 * The transport selected by `EMBED_TRANSPORT`, or by `id` when the caller knows
 * which one it needs (retrieval reads it off the index envelope). Resolved per
 * call rather than cached in a module constant, so a test or a script can set the
 * env var and get the transport it asked for without fighting module-load order.
 */
export function getEmbedTransport(id?: string): EmbedTransport {
  return resolveTransport(id);
}

/**
 * `id` overrides EMBED_TRANSPORT for one call (AD-2, 30/08/2026).
 *
 * A QUESTION must be embedded by the model that built the INDEX it is searching,
 * and that is a property of the index, not of the current configuration. Without
 * this, one global setting forces every collection onto the same model, so
 * changing the default silently invalidates every index not rebuilt with it —
 * and rebuilding is not always cheap: this account's Titan quota is 60 calls per
 * minute, which puts the AWS collections at ~10 hours.
 *
 * So ingest still writes with the configured default (a new index should use the
 * current choice), while retrieval reads the model NAME off the index it loaded
 * and asks for that transport by id. Collections built with different models
 * coexist correctly instead of one of them quietly returning nonsense.
 */
function resolveTransport(id?: string): LocalTransport {
  const key = id ?? EMBED_TRANSPORT;
  const transport = TRANSPORTS[key];
  if (!transport) {
    throw new Error(
      `מנוע הטמעה לא מוכר: "${key}". ` +
      `הערכים האפשריים: ${Object.keys(TRANSPORTS).join(", ")}.`,
    );
  }
  return transport;
}

// ---------------------------------------------------------------------------
// The retry loop
// ---------------------------------------------------------------------------

/**
 * Two defences against a low rate limit, both learned by hitting it:
 *   - EMBED_PACE_MS (applied by `embedAll` between batches): a deliberate pause,
 *     so we stay under the limit instead of discovering it by failing.
 *   - exponential backoff up to ~64s here, for when we are throttled anyway.
 *
 * Eight attempts is 1+2+4+…+64 ≈ two minutes of waiting, which is the right
 * order of magnitude for a per-minute quota and far too long for anything else —
 * hence `classifyError` short-circuiting the two cases where waiting cannot help.
 */
async function embedWithRetry(
  transport: LocalTransport,
  texts: string[],
  purpose: "document" | "query",
  attempts = 8,
): Promise<EmbedResult> {
  for (let i = 0; i < attempts; i++) {
    const attemptFrom = Date.now();
    try {
      return await transport.embed(texts, purpose);
    } catch (err: any) {
      const kind = classifyError(err);

      if (kind === "daily-ceiling") {
        throw new Error(
          `המכסה היומית של ${transport.modelId} נוצלה ("${err?.message ?? ""}"). ` +
          `זו לא חריגה זמנית — המתנה לא תעזור היום. ` +
          `מכסות Bedrock הן פר-מודל, אז מודל אחר חסום לא אומר ש-Bedrock סגור: ` +
          `החלף EMBED_TRANSPORT למודל אחר, או המתן ליום הבא.`,
          { cause: err },
        );
      }
      // Not a capacity problem (permissions, bad model id, malformed body) —
      // or we are out of attempts. Either way, retrying is not the answer.
      if (kind !== "throttle" || i === attempts - 1) throw err;
      recordRejectedCall("embed", Date.now() - attemptFrom, (err as Error)?.name ?? "unknown");

      const waitMs = Math.min(1000 * 2 ** i, 64_000);
      const waitedFrom = Date.now();
      process.stdout.write(`   נחסם זמנית (throttling) → ממתין ${waitMs / 1000} שניות\n`);
      await sleep(waitMs);
      recordRetryWait("embed", Date.now() - waitedFrom, (err as Error)?.name ?? "unknown");
    }
  }
  throw new Error("unreachable");
}

// ---------------------------------------------------------------------------
// The public entry point
// ---------------------------------------------------------------------------

/** How many overflow warnings to print before switching to a single summary. */
const MAX_OVERFLOW_WARNINGS = 5;

/**
 * Embed any number of texts, in order, with batching, pacing, retries and a
 * running token tally. The returned `vectors` array is index-aligned with the
 * input — callers rely on that to attach each vector back to its chunk.
 *
 * `onProgress(done, total)` is called after every batch so a long ingest can
 * print progress and checkpoint its work; it is optional because `ask.ts` embeds
 * one query and has nothing to report.
 */
export async function embedAll(
  texts: string[],
  purpose: "document" | "query",
  onProgress?: (done: number, total: number) => void,
  transportId?: string,
): Promise<EmbedResult> {
  if (texts.length === 0) return { vectors: [], tokens: 0 };

  // `transportId` is how retrieval pins a query to the model that built the index
  // it is about to search. Omitted everywhere else, which keeps ingest on the
  // configured default. See `resolveTransport`.
  const transport = resolveTransport(transportId);

  // Two limits, and the smaller one wins: EMBED_BATCH is our own politeness
  // setting, maxBatch is the provider's hard ceiling (96 for Cohere, 1 for
  // Titan). Taking the minimum is what lets the same call site work for both.
  const batchSize = Math.max(1, Math.min(EMBED_BATCH, transport.maxBatch));

  // Both size checks run over the WHOLE input before the first API call, so an
  // oversized chunk fails at second zero instead of at 90% of a paid ingest.
  let overflowCount = 0;
  for (let i = 0; i < texts.length; i++) {
    // Hard ceiling first. This one is a 400 from the API, not a truncation, so
    // there is nothing to warn about — the run cannot succeed as configured, and
    // CHUNK_CHARS is the single value that fixes it.
    if (texts[i].length > transport.maxInputChars) {
      throw new Error(
        `קטע ${i} באורך ${texts[i].length} תווים, ו-${transport.modelId} דוחה כל טקסט מעל ` +
        `${transport.maxInputChars} תווים (ValidationException, לא חיתוך שקט). ` +
        `הקטן את CHUNK_CHARS מתחת ל-${transport.maxInputChars} והרץ ingest מחדש. ` +
        `עוצרים לפני הקריאה הראשונה כדי לא לשלם על ריצה שתיפול באמצע.`,
      );
    }

    // Soft ceiling second. `truncate:"END"` means an over-window text SUCCEEDS
    // and returns a vector representing only part of the chunk. Nothing
    // downstream can detect that, so this warning is the only place it is visible.
    const estimated = estimateTokens(texts[i]);
    if (estimated <= transport.maxInputTokens) continue;
    overflowCount++;
    if (overflowCount <= MAX_OVERFLOW_WARNINGS) {
      console.warn(
        `אזהרה: קטע ${i} מוערך ב-~${estimated} טוקנים, וחלון הקלט של ${transport.modelId} ` +
        `הוא ${transport.maxInputTokens}. הסוף ייחתך (truncate) והווקטור ייצג רק חלק מהטקסט. ` +
        `הקטן את CHUNK_CHARS.`,
      );
    }
  }
  if (overflowCount > MAX_OVERFLOW_WARNINGS) {
    console.warn(`אזהרה: בסך הכל ${overflowCount} קטעים חורגים מחלון הקלט (הוצגו ${MAX_OVERFLOW_WARNINGS} ראשונים).`);
  }

  const vectors: number[][] = [];
  let tokens = 0;

  for (let start = 0; start < texts.length; start += batchSize) {
    const batch = texts.slice(start, start + batchSize);
    const result = await embedWithRetry(transport, batch, purpose);

    // The provider must return exactly one vector per input text, in order. If it
    // ever does not, every chunk after this point would be paired with the wrong
    // vector — a corruption that produces no error, only bad answers.
    if (result.vectors.length !== batch.length) {
      throw new Error(
        `${transport.id} החזיר ${result.vectors.length} וקטורים עבור ${batch.length} טקסטים. ` +
        `חוסר התאמה כזה היה משייך כל קטע לווקטור הלא נכון, ולכן עוצרים כאן.`,
      );
    }

    vectors.push(...result.vectors);
    tokens += result.tokens;
    onProgress?.(vectors.length, texts.length);

    // Pace only BETWEEN batches — a trailing sleep after the last one buys
    // nothing and, over a full ingest, is pure added wall-clock time.
    if (start + batchSize < texts.length && EMBED_PACE_MS > 0) await sleep(EMBED_PACE_MS);
  }

  return { vectors, tokens };
}

/**
 * One text in, one vector out. Used by the query path, which has exactly one
 * question to embed and no use for a batch. When the token count matters (the
 * cost readout), call `embedAll` and read `tokens` instead.
 */
export async function embedOne(text: string, purpose: "document" | "query"): Promise<number[]> {
  const { vectors } = await embedAll([text], purpose);
  return vectors[0];
}
