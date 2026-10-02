/**
 * types.ts — the shared contract every module builds against.
 *
 * Why this file exists: the same core logic runs in two places (a Lambda in the
 * cloud and a Node server on the laptop) and over two very different corpora
 * (Hebrew TAKAM instructions, English AWS docs). One set of shapes keeps those
 * four combinations from drifting apart. Nothing here imports anything — it is
 * pure description, so every other module can depend on it safely.
 */

/** One retrievable unit: a piece of text plus enough metadata to cite it. */
export interface Chunk {
  /** Globally unique within a collection: `${code}#${index}`. */
  id: string;
  /** Which corpus this belongs to: "takam", "aws-bedrock", "aws-lambda", ... */
  collection: string;
  /** Citation handle. TAKAM: instruction number ("7.6.1"). AWS: page slug. */
  code: string;
  /** Human-readable source title, shown to the user in the sources box. */
  title: string;
  /** Position of this chunk inside its document. */
  index: number;
  text: string;
  /**
   * Official keywords from the TAKAM catalogue API, when the document has them
   * (340 of 416 instructions do). They ride along in the chunk metadata because
   * they carry the formal vocabulary a citizen would never guess — "ערבויות
   * וביטחונות" for what a user calls "ערבות דיגיטלית".
   */
  keywords?: string[];
  /**
   * What KIND of document this passage came from, as the ministry classifies it
   * ("Instruction", "Form", "Message", "Tender Notice", "Attached File").
   *
   * It exists because the corpus stopped being instructions only. Before that
   * every TAKAM passage was announced to the model as "מתוך הוראה X", which was
   * true when the corpus held 411 instructions and became false for the 546
   * forms, messages and annexes added alongside them. Citing a fillable form as
   * though it were binding policy is exactly the kind of error a reader who
   * knows the regulations catches first, so the type travels with the passage.
   *
   * Absent for the AWS collections, which have no such distinction.
   */
  doctype?: string;
  /** Link back to the original document, for the sources box. */
  sourceUrl?: string;
  /**
   * One generated sentence saying where this passage sits — see
   * scripts/gen-context.ts.
   *
   * It joins what gets EMBEDDED and INDEXED and never what the answering model
   * reads. That split is the point: retrieval needs to know that a paragraph
   * reading "השיעור יהיה 70%" belongs to the toll-road instruction, while an
   * answer that quoted a generated sentence would be citing a machine, not a
   * regulation.
   */
  context?: string;
  /** Filled in by the ingest pass. Absent means "not embedded yet". */
  embedding?: number[];
}

/**
 * One collection's vector index — the whole "vector database" of this POC,
 * as a single serialisable object. Written to S3 in cloud mode, to disk locally.
 * The envelope records HOW it was built, because vectors from two different
 * embedding models are not comparable and mixing them silently corrupts search.
 */
export interface VectorIndex {
  collection: string;
  /** Exact model id, e.g. "cohere.embed-multilingual-v3". */
  embedModel: string;
  /** Which transport produced it, e.g. "bedrock:cohere". */
  embedTransport: string;
  region: string;
  /** Vector length. A mismatch against the query vector is a hard error. */
  dims: number;
  /**
   * Whether the generated per-chunk context sentences were part of what was
   * embedded.
   *
   * It is on the ENVELOPE, next to the model name, for the same reason: it
   * describes the vectors, and vectors built with a different input are not
   * comparable to ones built without it. Without this field the reuse check was
   * fooled — gen-context writes its sentences onto the stored chunks, so a
   * re-ingest compared a chunk-with-context against a chunk-with-context, found
   * them identical, and kept vectors that had been computed before any context
   * existed. Measured on chapter 9: 43 of 43 vectors byte-identical after a run
   * that was supposed to rebuild them all.
   */
  contextual?: boolean;
  /** ISO timestamp — surfaced in every answer so the user sees the corpus date. */
  indexedAt: string;
  chunks: Chunk[];
}

/** A chunk plus its similarity to the question. */
export interface Hit extends Chunk {
  score: number;
  /**
   * True when this passage was added by the LEXICAL table assist rather than by
   * semantic similarity — see retrieve.ts.
   *
   * It exists to keep two decisions apart. A bare rate table has almost no
   * semantic surface, so a phrase match is the only way to find it; but a
   * phrase match is also weak evidence that the corpus covers the question at
   * all. An earlier version expressed the assist by raising `score` to just
   * above the refusal threshold, and because the gate reads that same number,
   * the system began answering 6 of 10 questions it has no business answering.
   *
   * So the assist now adds CONTEXT and never votes on confidence: `score` stays
   * the true cosine, and `isConfident` ignores any hit carrying this flag.
   */
  lexical?: boolean;
  /**
   * Position in the fused ranking, when hybrid retrieval produced one.
   *
   * It has to travel with the hit because `score` cannot carry it: a chunk
   * found only by BM25 keeps its true (low) cosine, so any later sort by score
   * silently discards the fusion. That is exactly what happened — RRF was
   * computed correctly, then `diversify` re-sorted by score at the end and
   * threw the whole result away, and the named-instruction lookup appeared to
   * do nothing.
   */
  fusedRank?: number;
  /**
   * The user named this document explicitly ("what does instruction 7.6.1
   * say"). The cheap score floor is skipped for it — see isConfident.
   */
  named?: boolean;
  /**
   * When parent vote collapsed several chunks of one document into this hit,
   * the ids of ALL of them, this one included.
   *
   * It has to travel with the hit for the same reason fusedRank does: nothing
   * else can carry it. withParentDocuments builds its excerpt window around the
   * chunks that MATCHED, and it identifies them by id. Collapse the siblings
   * without recording them and a long instruction would be excerpted around one
   * chunk instead of three — the vote would improve the ranking and quietly
   * degrade the context the answer is written from.
   */
  merged?: string[];
  /** How many chunks of this document were in the candidate list. Diagnostic. */
  parentVotes?: number;
}

/** What one embedding call returns. Batch-shaped: one vector per input text. */
export interface EmbedResult {
  vectors: number[][];
  /** Input tokens consumed, for the cost readout. 0 when the provider omits it. */
  tokens: number;
}

/**
 * The embedding contract. Every transport (Bedrock/Cohere, Bedrock/Titan,
 * Gemini) implements exactly this, which is what makes the switch a config
 * value rather than a rewrite.
 */
export interface EmbedTransport {
  /** Transport id as it appears in EMBED_TRANSPORT. */
  readonly id: string;
  readonly modelId: string;
  readonly region: string;
  readonly dims: number;
  /** Max texts accepted in a single call. */
  readonly maxBatch: number;
  /**
   * `purpose` matters for Cohere, which embeds a stored passage and a search
   * query differently. Providers that ignore the distinction just discard it.
   */
  embed(texts: string[], purpose: "document" | "query"): Promise<EmbedResult>;
}

/** What one generation call returns, normalised across providers. */
export interface GenerateResult {
  modelId: string;
  text: string;
  inputTokens: number;
  outputTokens: number;
  /**
   * Why the model stopped: "stop" when it finished its answer, "length" when it
   * hit the output cap, or whatever else the provider reported.
   *
   * This field exists because its absence hid a real defect for weeks. Every
   * provider returns a stop reason and no transport read it, so an answer cut
   * off at the token limit was returned with a full usage block and a sources
   * list — indistinguishable from a complete one. The reader saw a short answer
   * and concluded the system had missed something, when in fact it had been
   * interrupted mid-sentence. A truncated answer must never again be able to
   * present itself as a finished one.
   */
  stopReason?: string;
}

/**
 * The generation contract. Bedrock and OpenRouter both reduce to this, so
 * `answer.ts` never knows which one answered.
 */
export interface LlmTransport {
  readonly id: string;
  readonly modelId: string;
  generate(system: string, user: string): Promise<GenerateResult>;
  /**
   * The same generation, delivered a piece at a time.
   *
   * OPTIONAL on purpose. A transport that cannot stream simply omits it, and
   * `ask` falls back to `generate` — so adding streaming to one provider never
   * breaks the others, which is the whole point of AD-2.
   *
   * Why it exists at all (measured 03/09/2026): of the 11.6 seconds one
   * question took, 10.5 were this call. Retrieval was ~1 second. There is no
   * faster model to switch to — Haiku already IS the fast one — so the only
   * honest way to shorten the wait is to stop making the reader wait for the
   * LAST token before showing the FIRST. Measured first token: 1.3 seconds.
   *
   * The resolved value is the identical `GenerateResult` the blocking call
   * returns, so everything downstream — citation checking, cost, sources —
   * runs on exactly the same object and cannot drift between the two paths.
   */
  generateStream?(
    system: string,
    user: string,
    onToken: (delta: string) => void,
  ): Promise<GenerateResult>;
}

/** The full answer handed back to the UI. */
export interface Answer {
  text: string;
  /**
   * Present when the system is asking WHICH TOPIC the user meant instead of
   * answering. `text` then holds the prompt and `sources` is empty: nothing has
   * been answered yet, so nothing may be cited.
   */
  clarify?: import("./clarify.js").Clarification;
  /**
   * true when `text` is a refusal — the deterministic gate's sentence, or a
   * model-written answer that OPENS with it. Decided exactly once, in answer.ts,
   * after the full text is known; every consumer reads this field instead of
   * re-matching the text.
   *
   * Why a field and not a function: the same question used to be answered by
   * text matching in four separate copies — answer.ts, public/index.html,
   * scripts/eval-battery.mjs, and lambda/query.ts, which had already drifted to
   * `sources.length === 0`. That copy would have logged every refusal-with-quote
   * as answered the moment REFUSAL_KEEPS_QUOTED_SOURCES was switched on. Text
   * matching broke the measurement three times (a sentence continued past its
   * full stop, a markdown heading, bold emphasis); each fix had to be mirrored
   * by hand, and a mirror that is kept by hand is eventually not kept.
   *
   * Required, not optional, so the compiler lists every return site that forgot.
   * A clarification is not a refusal: nothing was declined, a choice was offered.
   */
  refused: boolean;
  /**
   * How much of this answer's elapsed time was spent asleep in a backoff,
   * waiting for the account's rate quota, and how many backoffs that was.
   * Written once by ask(), from the retry ledger (retry-meter.ts); always
   * present, 0 when nothing was throttled.
   *
   * WHY IT IS ON THE ANSWER. The latency target is "first token within 6
   * seconds" and the gate run of 15/09/2026 measured 9.1s at p90 - with 503
   * throttles across 400 questions folded into the same number. Work and
   * waiting were indistinguishable, so the target could not be read as passed
   * or failed. A consumer that wants the old meaning still has it: elapsed
   * time is unchanged, and this is what to subtract from it.
   */
  retryWaitMs?: number;
  retryCount?: number;
  /**
   * Time spent in calls that were sent and rejected - the round trip of a
   * throttled request, which bought nothing. Separate from retryWaitMs because
   * a different thing is being paid for: one is a deliberate pause, the other
   * is a call that failed after taking real time. Both are the quota; neither
   * is the assistant being slow.
   */
  retryRejectedMs?: number;
  retryRejectedCalls?: number;
  /** The same, per stage, so "which model is being throttled" is answerable. */
  retryByStage?: { [stage: string]: { waitMs: number; retries: number; rejectedMs: number; rejectedCalls: number } };
  /**
   * How many generation calls this answer sent to each region (GEN_REGIONS,
   * 20/09/2026). Evidence that the rotation ran, carried per row rather than
   * as a run total: a rotation that quietly collapsed to one region would leave
   * the latency where it was and read as "spreading does not help".
   */
  genRegions?: Record<string, number>;
  /** Only the instructions actually cited in `text`. */
  sources: { code: string; title: string; score: number; sourceUrl?: string; doctype?: string }[];
  /** All retrieved passages, for the "show your work" panel. */
  retrieved: { code: string; title: string; score: number }[];
  /**
   * Which parts of the corpus this answer actually searched.
   *
   * It is reported on every answer because the most damaging defect this
   * project has had was invisible narrowing: the assistant searched 18
   * instructions out of 411 for weeks, answered "I did not find it" in a
   * perfectly ordinary voice, and nothing anywhere said the search had been
   * smaller than the corpus. A collection whose index is missing is dropped
   * silently by resolveMembers, so "how much did you look at" has to travel
   * with the answer rather than being inferable from it.
   */
  coverage: { searched: number; declared: number; missing: string[] };
  /** true when the reranker (rerank.ts) reordered the candidates before generation; absent/false when it was off or fell back. */
  reranked?: boolean;
  /**
   * How many documents parent vote promoted for this question; absent when the
   * flag is off, 0 when it ran and nothing qualified.
   *
   * It is on the answer rather than in a log line because of rule 2: a metric
   * that did not move must be separable into "did not help" and "did not run",
   * and only per-question evidence tells those apart.
   */
  parentVoted?: number;
  /**
   * Why the reranker did not run, when it did not: "off", "nothing-to-rank",
   * "unparseable", or the error that ended it. Absent when it ran.
   *
   * This field exists because its absence cost a whole baseline: the ranked
   * list was identical to the search list in 226 of 300 questions and no
   * output anywhere said so.
   */
  rerankReason?: string;
  /**
   * What REFUSAL_QUOTE_CHECK decided, on a refusal that passed the quote test:
   * "answers" kept the sources, "explains" cleared them, "failed" means the
   * call did not return a verdict and the sources were cleared (fail closed).
   * Absent when the check had nothing to check or the flag is off.
   *
   * Per answer, for the same reason as rerankReason: a component that can be
   * skipped silently needs evidence that it ran before its metric is read.
   */
  refusalQuoteCheck?: "answers" | "explains" | "failed";
  /**
   * ANSWER_SCOPE_FIRST evidence: true when the system prompt actually sent for
   * this answer carried the scope-first rule. Absent when the flag is off or the
   * model was never called (gate refusal, clarification). run-eval invalidates a
   * run where the flag is on and a model-written answer lacks `true`. 22/09/2026.
   */
  scopeFirstRule?: boolean;
  /**
   * Citation numbers UNVERIFIED_ALLOWS_PASSAGE_REFS removed from `unverified`
   * because a passage shown to the model names them. Absent when the flag is
   * off; an empty array means the rule ran and cleared nothing. Evidence field,
   * like refusalQuoteCheck.
   */
  unverifiedCleared?: string[];
  /**
   * SOURCES_ADD_QUOTED_DOC evidence: codes of the documents an answer quoted
   * from without naming them, which the rule added to `sources`. Absent when the
   * flag is off or the answer is a refusal; an empty array means the rule ran
   * and added nothing. Added 16/09/2026.
   */
  quotedDocsAdded?: string[];
  /**
   * PARENT_TOP_DOC_SLACK evidence. Absent when the flag is off; null when it ran
   * and the top document fit the budget or overflowed by more than the slack;
   * the document's code when the slack let it in whole. Same reason as
   * unverifiedCleared: a rule that fires rarely needs proof it ran at all.
   */
  parentSlackDoc?: string | null;
  /**
   * The passages in the order the reranker left them, before they were widened
   * into whole instructions.
   *
   * `retrieved` above is captured BEFORE stage 1½, so it is the search's twelve
   * candidates and answers "did the search find it at all" — recall@12. Nothing
   * on the answer used to expose the order the model actually read in, which
   * meant top-1 and top-3 after reranking could only be measured by running the
   * reranker a SECOND time from a benchmark script. That is a different model
   * call on the same question: twice the cost, and a different measurement
   * dressed up as the same one.
   *
   * Pure observability — the answer is byte-for-byte what it was without this
   * field. It exists so the accuracy runner reads what happened instead of
   * re-enacting it.
   */
  ranked?: { code: string; title: string; score: number }[];
  /**
   * Citation numbers that appear in `text` but match no retrieved passage.
   * Normally empty. When it is not, the answer is shown with an explicit
   * warning: a model that will not invent a fact can still invent the label
   * attached to one, and a wrong instruction number is the first error a
   * reader who knows the regulations will notice.
   */
  unverified: string[];
  /**
   * True when the model hit GEN_MAX_TOKENS and stopped mid-answer. The stop
   * reason was already read by every transport (see GenerateResult.stopReason)
   * but never reached the Answer, so the UI could not warn about it: found
   * 02/10/2026 on the demo question about tender exemptions, whose answer
   * stopped mid-word and looked complete.
   */
  truncated?: boolean;
  /**
   * True when the deterministic confidence gate (MIN_SCORE) refused before any
   * model call. Such an answer never reaches the reranker, by design, so the
   * eval must not count it as "a configured stage that did not run". Added
   * 02/10/2026: on the 15-instruction set one gold question fell below the gate
   * and the run declared itself invalid for a skip that was the gate working.
   */
  refusedAtGate?: boolean;
  collection: string;
  model: string;
  /** Which transport answered, e.g. "openrouter" or "bedrock:deepseek". */
  route: string;
  indexedAt: string;
  usage: { inputTokens: number; outputTokens: number; costUSD: number };
}

/** Request shape accepted by both the local server and the Lambda Function URL. */
export interface AskRequest {
  question: string;
  collection?: string;
}
