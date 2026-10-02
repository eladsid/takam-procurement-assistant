/**
 * rerank.ts — the "question track", stage 1½: put the candidates in the right order.
 *
 * Why this stage exists, in one number pair. After the whole-corpus search
 * (retrieve.ts, retrieveUnion) the correct instruction is somewhere in the top
 * three for 82% of citizen-phrased questions, but FIRST for only 54%. The
 * search finds the right document and then loses the coin toss between it and
 * its neighbours, because a cosine and a BM25 score both measure "shares
 * meaning / shares words with the question", not "contains the answer".
 *
 * A reranker reads the question and each candidate passage TOGETHER and asks
 * only that last question. That is a different, more expensive computation —
 * it cannot run over 28,514 chunks, but it does not have to: it runs over the
 * twelve the search already picked. Search casts the net; the reranker sorts
 * the catch.
 *
 * Implementation choice: the generation model (Haiku) as the reranker, rather
 * than a dedicated rerank model. Bedrock offers Cohere Rerank, but this account
 * has already had model access refused after a signed agreement, and a
 * reranker that may not exist tomorrow is not a foundation. Haiku is already
 * authorised, already wired, and the cost is one short call per question. The
 * interface below is model-agnostic so a dedicated reranker can replace the
 * body without touching the callers.
 *
 * Measured 07/09/2026, 76 citizen-phrased questions, the same twelve
 * candidates with and without this stage: correct instruction first 43 -> 56
 * (57% -> 74%; 16 gained, 3 lost), in the top three 58 -> 68 (76% -> 89%).
 * $0.0042 and ~4.5 seconds per question. Six questions had the right
 * instruction outside the twelve candidates altogether — that is the search's
 * ceiling, and no reranker reaches past it.
 *
 * Failure policy: the reranker is an optimisation, the answer is the product.
 * Any error, timeout or unparseable reply returns the candidates exactly as
 * they arrived, and says so in `applied`, so a throttled afternoon degrades to
 * "yesterday's ranking" rather than to "no answer".
 */
import { RERANK, TOP_K } from "./config.js";
import type { Hit } from "./types.js";
import { recordRetryWait, recordRejectedCall } from "./retry-meter.js";

const SYSTEM = `אתה מדרג קטעים מהוראות תכ"ם (תקנון כספים ומשק) לפי הסיכוי שהם מכילים את התשובה לשאלה.
תקבל שאלה ורשימת קטעים ממוספרים. החזר מערך JSON בלבד של מספרי הקטעים, מהרלוונטי ביותר לפחות רלוונטי, עד 5 מספרים.
קטע רלוונטי הוא קטע שמכיל את התשובה עצמה או את הכלל שממנו נגזרת התשובה — לא קטע שרק חולק מילים עם השאלה.
אל תוסיף טקסט, הסבר או סימני קוד. דוגמה לפלט תקין: [3, 1, 7]`;

/** How much of each passage the reranker sees. Enough to judge, cheap enough to send twelve. */
const PASSAGE_CHARS = 700;
/** Hard budget for ONE rerank attempt. Measured median is ~4.5s. */
const RERANK_TIMEOUT_MS = 15_000;

/**
 * How many times a rerank call is retried before the search order answers.
 *
 * This exists because of a measured, silent failure. Until now this module
 * called the transport ONCE, with no retry, while the answer itself went
 * through generateWithRetry with exponential backoff. Bedrock throttles under
 * eval load, the catch-all below turned every throttle into "the reranker was
 * not applied", and nothing said so. On the 08/09 baseline run the ranked list
 * was byte-identical to the search list in 226 of 300 questions: the stage the
 * whole accuracy plan leans on ran in one question out of four. Where it did
 * run, top-1 was 82.4%; where it did not, 74.3%.
 *
 * Three attempts with the same backoff the answer path uses, and a reason on
 * the way out so a future run can never hide this again.
 */
const RERANK_ATTEMPTS = 5;

/**
 * First backoff step, doubling from there: 1s, 2s, 4s, 8s. Fifteen seconds of
 * patience in the worst case.
 *
 * MEASURED 10/09/2026, and this is the second correction to this block in one
 * day. Three attempts at 500ms/1000ms recovered most of the stage — the 08/09
 * baseline ranked 74 of 300 questions, the run after the retry landed on 241 —
 * but 58 questions still lost it to ThrottlingException. The full run took 413
 * throttles across 119 minutes at concurrency 1, roughly one every seventeen
 * seconds, and a budget of 1.5 seconds of total waiting is simply too small
 * against pressure at that rate.
 *
 * The reranker can afford to wait where the answer cannot: it has a working
 * fallback (the search order) and no user is watching it. And since a run is
 * now INVALIDATED when this stage does not run, a fast failure here is not
 * cheap at all — it costs the whole twelve-dollar measurement.
 */
const RERANK_BACKOFF_MS = 1_000;

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms));

export interface RerankResult {
  hits: Hit[];
  /** false when the reranker was off, had nothing to do, or failed — the hits are then untouched. */
  applied: boolean;
  /**
   * Why it was not applied: "off", "nothing-to-rank", "unparseable", or the
   * error name that ended it ("ThrottlingException", "rerank timeout"...).
   * Absent when it WAS applied. A failure that does not say its name is how
   * this stage stayed dead for a whole baseline run.
   */
  reason?: string;
  ms: number;
  inputTokens: number;
  outputTokens: number;
}

const untouched = (hits: Hit[], reason: string): RerankResult =>
  ({ hits, applied: false, reason, ms: 0, inputTokens: 0, outputTokens: 0 });

export async function rerank(question: string, hits: Hit[]): Promise<RerankResult> {
  if (!RERANK) return untouched(hits, "off");

  // Table-assist rows are shown, never ranked (see retrieve.ts); a document the
  // user asked for BY NUMBER stays in front no matter what the reranker thinks
  // of its opening paragraphs. Only the rest is up for reordering.
  const extras = hits.filter(h => h.lexical);
  const named = hits.filter(h => !h.lexical && h.named);
  const cand = hits.filter(h => !h.lexical && !h.named);
  if (cand.length < 2) return untouched(hits, "nothing-to-rank");

  const t0 = Date.now();
  // Imported at call time, not at module load: answer.ts imports this module,
  // so a top-level import here would close a cycle (same reason as query-prep).
  const { getLlmTransport, isRetryable } = await import("./answer.js");

  let lastReason = "unknown";

  for (let attempt = 1; attempt <= RERANK_ATTEMPTS; attempt++) {
    const attemptFrom = Date.now();
    try {

    const listing = cand
      .map((h, i) => `[${i + 1}] הוראה ${h.code} — ${h.title}\n${h.text.replace(/\s+/g, " ").slice(0, PASSAGE_CHARS)}`)
      .join("\n\n");
    /**
     * Bounded wait. Measured 07/09/2026: a momentary DNS failure
     * (getaddrinfo ENOTFOUND bedrock-runtime) turned one question into a
     * 35-minute hang, because nothing between here and the socket had a clock.
     * The reranker is the one stage that has a perfectly good fallback — the
     * search order — so it gets a hard budget. The underlying call is not
     * cancelled (the transport has no abort handle yet); the answer simply
     * stops waiting for it.
     */
    const res = await Promise.race([
      getLlmTransport().generate(SYSTEM, `השאלה: ${question}\n\nהקטעים:\n\n${listing}`),
      new Promise<never>((_, reject) =>
        setTimeout(() => {
          // Named TimeoutError so isRetryable() treats it as capacity and the
          // next attempt runs; a bare Error would end the stage on the spot.
          const e = new Error("rerank timeout");
          e.name = "TimeoutError";
          reject(e);
        }, RERANK_TIMEOUT_MS),
      ),
    ]);

    // The first JSON array in the reply, nothing else. Numbers outside the
    // candidate range and repeats are dropped rather than trusted.
    const match = String(res.text ?? "").match(/\[[\d,\s]*\]/);
    const order = match
      ? (JSON.parse(match[0]) as unknown[]).map(Number).filter(i => Number.isInteger(i) && i >= 1 && i <= cand.length)
      : [];
    if (!order.length) return untouched(hits, "unparseable");

    const seen = new Set<number>();
    const ranked: Hit[] = [];
    for (const i of order) { if (!seen.has(i)) { seen.add(i); ranked.push(cand[i - 1]); } }
    // Anything the reranker did not mention keeps its search order, after the ranked ones.
    for (let i = 1; i <= cand.length; i++) if (!seen.has(i)) ranked.push(cand[i - 1]);

    // Back to the usual passage count: the caller fetched a wider net only so
    // the reranker had something to choose from. The named documents do not
    // count against it — they were asked for.
    return {
      hits: [...named, ...ranked.slice(0, TOP_K), ...extras],
      applied: true,
      ms: Date.now() - t0,
      inputTokens: res.inputTokens ?? 0,
      outputTokens: res.outputTokens ?? 0,
    };
    } catch (err) {
      /**
       * A throttle is capacity, not an answer: retry it the way the answer path
       * does. Anything else (a malformed request, a model that is not
       * authorised) will fail identically on the next attempt, so it stops
       * here. Either way the reason travels out, so a run can report how often
       * this stage did not run instead of quietly showing yesterday's ranking.
       */
      lastReason = (err as Error)?.name || String(err);
      // The failed call itself cost time; that is contention, not ranking work.
      recordRejectedCall("rerank", Date.now() - attemptFrom, lastReason);
      if (!isRetryable(err) || attempt === RERANK_ATTEMPTS) break;
      /**
       * Charged to the retry meter under "rerank". This stage is 68% of
       * everything that happens before the answer model is even called
       * (measured 18/09/2026), so a backoff here lands squarely inside the
       * user's wait for the first word - and used to be invisible in it.
       */
      const waitedFrom = Date.now();
      await sleep(RERANK_BACKOFF_MS * 2 ** (attempt - 1));
      recordRetryWait("rerank", Date.now() - waitedFrom, lastReason);
    }
  }

  return untouched(hits, lastReason);
}
