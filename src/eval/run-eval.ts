/**
 * run-eval.ts — one run, every number. `npm run eval`.
 *
 * WHAT THIS REPLACES
 * Until now each measurement was a throwaway script in a scratch folder, and
 * two of them disagreed because one of them silently skipped 16 of 50 requests
 * (`if (!r.ok) continue`). The numbers in this project's memlog are only worth
 * anything if the thing that produced them is the same thing every time and
 * counts what it drops. So: one entry point, one results file, explicit
 * counters, and a run that declares itself INVALID rather than reporting a
 * denominator it quietly shrank.
 *
 * WHY IT CALLS ask() IN-PROCESS INSTEAD OF OVER HTTP
 * The local server does not reload code (tsx without watch), and measuring
 * against a stale server has already cost this project two days. Importing the
 * core directly means the measurement always runs the code on disk, and it
 * removes HTTP as a source of failures that look like model failures.
 *
 * USAGE
 *   npm run eval                  # the full run: gold + controls
 *   npm run eval -- --limit=4     # smoke test, a handful of each
 *   npm run eval -- --gold-only   # or --controls-only
 */

import { writeFileSync, existsSync, unlinkSync, readFileSync, mkdirSync } from "node:fs";
import { DEFAULT_COLLECTION, GEN_MODELS, GEN_REGIONS, LLM_ROUTE, RERANK, REFUSAL_QUOTE_CHECK, UNVERIFIED_ALLOWS_PASSAGE_REFS, PARENT_TOP_DOC_SLACK, SOURCES_ADD_QUOTED_DOC, ANSWER_SCOPE_FIRST, JUDGE_WITH_QUESTION, JUDGE_QUESTION_POLAR_KIND, collectionDef, describeConfig } from "../core/config.js";
import { ask } from "../core/answer.js";
import { warmUnion } from "../core/retrieve.js";
import { isRefusal, mentionsRefusal, loadGold, loadControls, type GoldKind, type GoldRecord, type ControlRecord } from "./gold.js";
import { judgeAnswer, type Verdict } from "./judge.js";
import type { Answer } from "../core/types.js";
import { readRetryLedger } from "../core/retry-meter.js";

// ---------------------------------------------------------------- parameters

const args = process.argv.slice(2);
const flag = (n: string): string | undefined => args.find(a => a.startsWith(`--${n}=`))?.split("=")[1];
const has = (n: string): boolean => args.includes(`--${n}`);

const LIMIT = flag("limit") ? Number(flag("limit")) : 0;
/**
 * Continue a run that was declared invalid, asking ONLY the questions it lost.
 *
 * Why resume rather than re-run. The first full run answered 290 of 400 and
 * threw away 110 to throttling; re-running all 400 would pay for the 290 again
 * — about $8.50 — to learn nothing new about them. The lost questions were lost
 * for a reason unrelated to their content (they are spread evenly across the
 * set, positions 23 to 296, and across all fifteen chapters in proportion), so
 * the answers already collected are as valid as they would be on a second pass.
 *
 * The one thing that would invalidate a merge is the pipeline changing between
 * the two passes. Nothing under src/core changes here, and the config in effect
 * is written into both files so a later reader can check rather than trust.
 */
const RESUME = flag("resume");
/**
 * Measure a DIFFERENT gold file — the annex set, say — without disturbing the
 * baseline one. Records are never added to gold.json: changing its denominator
 * would silently invalidate every comparison drawn against it.
 */
const GOLD_FILE = flag("gold");
/**
 * The same for the control set: a subset file of controls.json records, so a
 * check that only matters on specific controls (c-044..c-096, beyond any
 * --limit slice) can be measured on exactly those. Added 14/09/2026.
 */
const CONTROLS_FILE = flag("controls");
/** Print what the run would ask and exit. Costs nothing, drives the real selection code. */
const PLAN = args.includes("--plan");
/**
 * Re-derive the refusal flags and the metric table from an existing results
 * file, using the CURRENT definitions and no model calls.
 *
 * It exists because a definition can be corrected after the answers were
 * collected — which is exactly what happened to isRefusal — and re-asking 400
 * questions to apply a string fix would be absurd. The answers are stored, so
 * the measurement can be recomputed for free; what must never happen is a file
 * whose rows were scored under two different definitions.
 */
const RECOMPUTE = flag("recompute");
/**
 * Re-run the JUDGE on rows whose judge call failed, using the stored answer.
 *
 * A judge failure is not a wrong verdict, it is a missing one — the question
 * was asked and answered, and only the adjudication fell over. Re-asking the
 * question would spend a whole answer to recover a judgement, and would also
 * replace an answer that is already part of the measurement. So the stored text
 * is judged again instead: two model calls rather than two questions.
 */
const REJUDGE = flag("rejudge");
const GOLD_ONLY = has("gold-only");
const CONTROLS_ONLY = has("controls-only");
/**
 * How many questions are in flight at once.
 *
 * Two, not more. Iron rule 1 forbids two measurement RUNS at a time because
 * they throttle each other; inside one run the same physics applies, just more
 * gently. Every throttle is counted separately below, and if the count is not
 * zero the right response is to drop this to 1 and re-run, not to accept the
 * numbers.
 */
const CONCURRENCY = Number(process.env.EVAL_CONCURRENCY ?? flag("concurrency") ?? 2);

/** Five attempts with a long backoff — see askTimed. */
const MAX_ATTEMPTS = Number(flag("attempts") ?? 5);

const LOCK_PATH = "eval/.lock";
const RESULTS_DIR = "eval/results";

// ------------------------------------------------------------------ counters

const counters = {
  attempted: 0,
  answered: 0,
  failed: 0,
  throttled: 0,
  /**
   * Retry-meter evidence (19/09/2026). `retryMeterRows` is how many answers
   * came back carrying a ledger at all - if it is not the number of answers,
   * the meter did not run and every wait number below is meaningless rather
   * than zero. The other three are the waiting itself.
   */
  retryMeterRows: 0,
  retryRowsWaited: 0,
  retryWaitTotalMs: 0,
  retryCountTotal: 0,
  /** Calls that were sent and turned away, and what they cost. Added 19/09/2026. */
  retryRejectedTotalMs: 0,
  retryRejectedCalls: 0,
  clarified: 0,
  judgeFailed: 0,
  /** Gold questions whose ranking stage actually ran. See rerankSkipped. */
  reranked: 0,
  /**
   * Answers where the structural flag (Answer.refused) disagrees with the text
   * test this runner scores refusals by (isRefusal).
   *
   * The runner deliberately keeps scoring by text: changing HOW refusals are
   * counted in the middle of the plan is what bent this number three times, and
   * every earlier result file was scored that way. So the flag is not trusted
   * here — it is checked. Expected 0. By construction the two can only differ
   * on an empty model reply (answer.ts counts that as refused, the text test
   * does not), which is itself worth seeing.
   */
  refusedFlagMismatch: 0,
  /** REFUSAL_QUOTE_CHECK verdicts, per kind - the evidence the check ran. */
  quoteCheckAnswers: 0,
  quoteCheckExplains: 0,
  quoteCheckFailed: 0,
  /**
   * Refusals that carry sources WITHOUT an "answers" verdict, while the check is
   * on. Expected 0 by construction; anything else means a path around the check
   * exists, and the run is invalid - the same rule as a reranker that skipped.
   */
  refusalSourcesUnchecked: 0,
  /**
   * UNVERIFIED_ALLOWS_PASSAGE_REFS: answers where the rule ran (flag on) and
   * answers where it removed at least one number from the warning. The first
   * proves the component executed; the second is what it did.
   */
  passageRefsRan: 0,
  passageRefsCleared: 0,
  /**
   * PARENT_TOP_DOC_SLACK: answers where the rule ran (flag on, model called) and
   * answers where it let the top document in whole. Added 15/09/2026.
   */
  topDocSlackRan: 0,
  topDocSlackUsed: 0,
  /**
   * SOURCES_ADD_QUOTED_DOC (16/09/2026): model-written answers the rule had to
   * run on (flag on, not a refusal, not a clarification), answers that carry its
   * evidence field, and answers where it added a document. Expected must equal
   * ran, or the run is invalid - a stage configured on that did not run.
   */
  quotedDocsExpected: 0,
  quotedDocsRan: 0,
  quotedDocsUsed: 0,
  /**
   * ANSWER_SCOPE_FIRST (22/09/2026): answers where the model was called with the
   * flag on (Answer.ranked is set only on that path), and answers whose SENT
   * prompt carried the rule. Expected must equal ran, or the run is invalid.
   */
  scopeFirstExpected: 0,
  scopeFirstRan: 0,
};
const refusedFlagMismatchIds: string[] = [];

/** Compare the structural flag with the text test, and remember any row where they part. */
function checkRefusedFlag(id: string, ans: Answer): void {
  if (ans.refused !== isRefusal(ans.text)) {
    counters.refusedFlagMismatch++;
    refusedFlagMismatchIds.push(id);
  }
}

/**
 * Count what REFUSAL_QUOTE_CHECK did on this answer, and catch a refusal whose sources bypassed it.
 * Also counts UNVERIFIED_ALLOWS_PASSAGE_REFS evidence - both are post-generation answer checks.
 */
function countQuoteCheck(ans: Answer): void {
  if (ans.unverifiedCleared) counters.passageRefsRan++;
  if (ans.unverifiedCleared?.length) counters.passageRefsCleared++;
  if (ans.parentSlackDoc !== undefined) counters.topDocSlackRan++;
  if (ans.parentSlackDoc) counters.topDocSlackUsed++;
  if (SOURCES_ADD_QUOTED_DOC && !ans.refused && !ans.clarify) counters.quotedDocsExpected++;
  if (ans.quotedDocsAdded) counters.quotedDocsRan++;
  if (ans.quotedDocsAdded?.length) counters.quotedDocsUsed++;
  if (ANSWER_SCOPE_FIRST && ans.ranked) counters.scopeFirstExpected++;
  if (ans.scopeFirstRule === true) counters.scopeFirstRan++;
  if (ans.refusalQuoteCheck === "answers") counters.quoteCheckAnswers++;
  if (ans.refusalQuoteCheck === "explains") counters.quoteCheckExplains++;
  if (ans.refusalQuoteCheck === "failed") counters.quoteCheckFailed++;
  if (REFUSAL_QUOTE_CHECK && ans.refused && ans.sources.length > 0 && ans.refusalQuoteCheck !== "answers") {
    counters.refusalSourcesUnchecked++;
  }
}

/**
 * Why the reranker did not run, counted by reason.
 *
 * Added 10/09/2026 after the 08/09 baseline was found to have been reranked in
 * only 74 of 300 questions — every failure swallowed by a catch-all, with the
 * search order silently standing in. A stage that can fail invisibly is a stage
 * whose measurement cannot be trusted, so the run now reports its own.
 */
const rerankSkipped: Record<string, number> = {};
const failureDetail: { q: string; reason: string }[] = [];

// ---------------------------------------------------------------------- rows

interface GoldRow {
  id: string;
  q: string;
  expect: string;
  chapter: number;
  fact: string;
  kind: GoldKind;
  /** Distinct instruction codes among the 12 candidates the search produced. */
  retrievedCodes: string[];
  /** Distinct instruction codes in reranked order — top-1 and top-3 come from here. */
  rankedCodes: string[];
  recallAt12: boolean;
  top1: boolean;
  top3: boolean;
  reranked: boolean;
  /** Refused by the confidence gate before any model call (so no rerank, by design). */
  refusedAtGate?: boolean;
  /** Why the ranking stage did not run, when it did not. Audit trail. */
  rerankReason?: string;
  /** REFUSAL_QUOTE_CHECK's verdict, when it ran. */
  refusalQuoteCheck?: Answer["refusalQuoteCheck"];
  /** The citation-warning numbers, and those UNVERIFIED_ALLOWS_PASSAGE_REFS cleared. Added 15/09/2026. */
  unverified?: string[];
  unverifiedCleared?: string[];
  /** PARENT_TOP_DOC_SLACK evidence (Answer.parentSlackDoc). Added 15/09/2026. */
  parentSlackDoc?: string | null;
  /** SOURCES_ADD_QUOTED_DOC evidence (Answer.quotedDocsAdded). Added 16/09/2026. */
  quotedDocsAdded?: string[];
  /** ANSWER_SCOPE_FIRST evidence (Answer.scopeFirstRule). Added 22/09/2026. */
  scopeFirstRule?: boolean;
  sources: string[];
  verdict: Verdict;
  answerText: string;
  firstTokenMs: number | null;
  /**
   * The part of firstTokenMs that was a backoff, and the whole answer's
   * backoff total. Added 19/09/2026 so the latency target stops being one
   * number that mixes writing with waiting for the account's rate quota.
   */
  firstTokenWaitMs: number;
  /**
   * The part of firstTokenMs spent inside calls that were rejected. Added a few
   * hours after firstTokenWaitMs, on 19/09/2026, because the first metered run
   * showed the split was still incomplete: rows that were never throttled came
   * out at 5.10s p90 while "work" on throttled rows read 8.64s, and the gap was
   * the round trip of calls that were turned away.
   */
  firstTokenRejectedMs?: number;
  retryWaitMs?: number;
  retryCount?: number;
  retryRejectedMs?: number;
  retryRejectedCalls?: number;
  retryByStage?: { [stage: string]: { waitMs: number; retries: number; rejectedMs: number; rejectedCalls: number } };
  /** Which regions served this row (GEN_REGIONS). Empty when the rotation is off. */
  genRegions?: Record<string, number>;
  totalMs: number;
  costUSD: number;
}

interface ControlRow {
  id: string;
  q: string;
  refused: boolean;
  /** The loose `includes` form the 07/09 baseline used, kept so the two can be compared. */
  refusedLoose: boolean;
  clarified: boolean;
  answerText: string;
  sources: string[];
  /** REFUSAL_QUOTE_CHECK's verdict, when it ran. */
  refusalQuoteCheck?: Answer["refusalQuoteCheck"];
  unverified?: string[];
  unverifiedCleared?: string[];
  parentSlackDoc?: string | null;
  quotedDocsAdded?: string[];
  scopeFirstRule?: boolean;
  /** Retry-meter fields, same meaning as on GoldRow. A refusal never streams,
   *  so it has no first-token time - but it can still have waited. */
  retryWaitMs?: number;
  retryCount?: number;
  retryRejectedMs?: number;
  retryRejectedCalls?: number;
  retryByStage?: { [stage: string]: { waitMs: number; retries: number; rejectedMs: number; rejectedCalls: number } };
  genRegions?: Record<string, number>;
  costUSD: number;
}

// ------------------------------------------------------------------- helpers

/**
 * Distinct document codes, in first-seen order.
 *
 * Table-assist rows are NOT filtered out, because by this point the answer only
 * carries {code, title, score} and the `lexical` flag is gone. That is the
 * honest behaviour anyway: those rows really were among the passages handed on,
 * so counting them is what "did the search surface the right document" means.
 */
const codesOf = (list: { code: string }[] | undefined): string[] => {
  const out: string[] = [];
  for (const h of list ?? []) if (!out.includes(h.code)) out.push(h.code);
  return out;
};

const percentile = (xs: number[], p: number): number => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * p))];
};

/**
 * Ask once, with retries, and time the first token.
 *
 * A refusal never streams (the model is not called at all), so its first-token
 * time is null rather than zero — averaging a zero in would make the latency
 * line look better every time the system declined to answer.
 */
async function askTimed(
  question: string,
): Promise<{
  ans: Answer;
  firstTokenMs: number | null;
  firstTokenWaitMs: number;
  firstTokenRejectedMs: number;
  totalMs: number;
} | null> {
  counters.attempted++;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const t0 = Date.now();
    let firstTokenMs: number | null = null;
    /**
     * How much of firstTokenMs was spent asleep in a backoff.
     *
     * Read from INSIDE the streaming callback, not from the finished answer:
     * the callback runs within the same async scope as ask(), so the ledger it
     * sees is the one for this question at the moment the first word arrived.
     * `ans.retryWaitMs` is the whole answer's waiting, which includes backoffs
     * that happened after the first token (the refusal quote check, a retry
     * mid-stream) and would over-subtract here.
     */
    let firstTokenWaitMs = 0;
    /** The same, for calls that were sent and rejected before this first word. */
    let firstTokenRejectedMs = 0;
    try {
      const ans = await ask(question, DEFAULT_COLLECTION, () => {
        if (firstTokenMs === null) {
          firstTokenMs = Date.now() - t0;
          const led = readRetryLedger();
          firstTokenWaitMs = led?.waitMs ?? 0;
          firstTokenRejectedMs = led?.rejectedMs ?? 0;
        }
      });
      counters.answered++;
      if (ans.retryWaitMs !== undefined) {
        counters.retryMeterRows++;
        counters.retryWaitTotalMs += ans.retryWaitMs;
        counters.retryCountTotal += ans.retryCount ?? 0;
        counters.retryRejectedTotalMs += ans.retryRejectedMs ?? 0;
        counters.retryRejectedCalls += ans.retryRejectedCalls ?? 0;
        if (ans.retryWaitMs > 0) counters.retryRowsWaited++;
      }
      return { ans, firstTokenMs, firstTokenWaitMs, firstTokenRejectedMs, totalMs: Date.now() - t0 };
    } catch (e) {
      const msg = (e as Error)?.message ?? String(e);
      if (/throttl|429|too many requests/i.test(msg)) counters.throttled++;
      if (attempt === MAX_ATTEMPTS) {
        counters.failed++;
        failureDetail.push({ q: question.slice(0, 60), reason: msg.slice(0, 140) });
        return null;
      }
      /**
       * Backoff sized to the measured ceiling, not to a habit.
       *
       * The first full run attempted about 21 requests a minute and the account
       * sustained roughly 10, so a throttled call needs to wait tens of seconds
       * for the window to clear, not the 1.5 and 6 seconds it waited before.
       * 110 of 400 questions were lost to giving up too early.
       */
      await new Promise(r => setTimeout(r, Math.min(45_000, 3_000 * 2 ** (attempt - 1))));
    }
  }
  return null;
}

/** Bounded-concurrency map that preserves input order in the output. */
async function pool<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return out;
}

const priceOf = (inTok: number, outTok: number): number => {
  const p = GEN_MODELS[LLM_ROUTE];
  return (inTok / 1e6) * p.priceIn + (outTok / 1e6) * p.priceOut;
};

// ------------------------------------------------------------------ the work

async function runGold(records: GoldRecord[]): Promise<GoldRow[]> {
  let evidenceShown = 0;

  const rows = await pool(records, CONCURRENCY, async (rec, i) => {
    const res = await askTimed(rec.q);
    if (!res) return null;
    const { ans, firstTokenMs, firstTokenWaitMs, firstTokenRejectedMs, totalMs } = res;

    if (ans.clarify) counters.clarified++;
    checkRefusedFlag(rec.id, ans);
    countQuoteCheck(ans);

    const verdict = await judgeAnswer(rec, ans);
    if (verdict.judgeFailed) counters.judgeFailed++;
    if (ans.reranked) counters.reranked++;
    else if (ans.rerankReason) rerankSkipped[ans.rerankReason] = (rerankSkipped[ans.rerankReason] ?? 0) + 1;

    const retrievedCodes = codesOf(ans.retrieved);
    const rankedCodes = codesOf(ans.ranked);

    /**
     * Evidence that the components ran, printed BEFORE any aggregate is shown.
     * Iron rule 2: a metric that does not move means "check the component ran"
     * first, and this is where that check is cheapest — 12 candidates in,
     * reranked true, sources out.
     */
    if (evidenceShown < 3) {
      evidenceShown++;
      console.log(
        `   [ראיה ${evidenceShown}] מועמדים=${retrievedCodes.length} · דורג=${ans.reranked ?? false} ` +
        `· אחרי דירוג=${rankedCodes.length} · מקורות=${ans.sources.length} · fact=${verdict.factMethod} · ` +
        `טוקן ראשון=${firstTokenMs ?? "—"}ms`,
      );
    }

    process.stdout.write(`\r   זהב ${i + 1}/${records.length}   `);

    return {
      id: rec.id,
      q: rec.q,
      expect: rec.expect,
      chapter: rec.chapter,
      fact: rec.fact,
      kind: rec.kind,
      retrievedCodes,
      rankedCodes,
      recallAt12: retrievedCodes.includes(rec.expect),
      top1: rankedCodes[0] === rec.expect,
      top3: rankedCodes.slice(0, 3).includes(rec.expect),
      reranked: ans.reranked ?? false,
      refusedAtGate: ans.refusedAtGate ?? false,
      rerankReason: ans.rerankReason,
      refusalQuoteCheck: ans.refusalQuoteCheck,
      unverified: ans.unverified,
      unverifiedCleared: ans.unverifiedCleared,
      parentSlackDoc: ans.parentSlackDoc,
      quotedDocsAdded: ans.quotedDocsAdded,
      scopeFirstRule: ans.scopeFirstRule,
      sources: ans.sources.map(s => s.code),
      verdict,
      answerText: ans.text,
      firstTokenMs,
      firstTokenWaitMs,
      firstTokenRejectedMs,
      retryWaitMs: ans.retryWaitMs,
      retryCount: ans.retryCount,
      retryRejectedMs: ans.retryRejectedMs,
      retryRejectedCalls: ans.retryRejectedCalls,
      retryByStage: ans.retryByStage,
      genRegions: ans.genRegions,
      totalMs,
      costUSD: ans.usage.costUSD + priceOf(verdict.usage.inputTokens, verdict.usage.outputTokens),
    } satisfies GoldRow as GoldRow;
  });

  process.stdout.write("\n");
  return rows.filter((r): r is GoldRow => r !== null);
}

async function runControls(records: ControlRecord[]): Promise<ControlRow[]> {
  const rows = await pool(records, CONCURRENCY, async (rec, i) => {
    const res = await askTimed(rec.q);
    if (!res) return null;
    const { ans } = res;
    if (ans.clarify) counters.clarified++;
    checkRefusedFlag(rec.id, ans);
    countQuoteCheck(ans);
    process.stdout.write(`\r   בקרות ${i + 1}/${records.length}   `);
    return {
      id: rec.id,
      q: rec.q,
      refused: isRefusal(ans.text),
      refusedLoose: mentionsRefusal(ans.text),
      clarified: Boolean(ans.clarify),
      answerText: ans.text,
      sources: ans.sources.map(s => s.code),
      refusalQuoteCheck: ans.refusalQuoteCheck,
      // Stored on controls too (15/09/2026): the first gate run with the flag
      // counted 3 cleared warnings and only 1 was traceable to a gold row.
      unverified: ans.unverified,
      unverifiedCleared: ans.unverifiedCleared,
      parentSlackDoc: ans.parentSlackDoc,
      quotedDocsAdded: ans.quotedDocsAdded,
      scopeFirstRule: ans.scopeFirstRule,
      retryWaitMs: ans.retryWaitMs,
      retryCount: ans.retryCount,
      retryRejectedMs: ans.retryRejectedMs,
      retryRejectedCalls: ans.retryRejectedCalls,
      retryByStage: ans.retryByStage,
      genRegions: ans.genRegions,
      costUSD: ans.usage.costUSD,
    } satisfies ControlRow as ControlRow;
  });
  process.stdout.write("\n");
  return rows.filter((r): r is ControlRow => r !== null);
}

// ---------------------------------------------------------------------- main

async function main(): Promise<void> {
  /**
   * A mode flag that no branch consumes must never fall through to a full run.
   *
   * This guard exists because it already happened: a patch silently failed to
   * insert the --rejudge branch, the flag parsed fine, and the process walked
   * past it into a complete 400-question measurement. It compiled, it took the
   * lock, and it printed nothing for ten minutes. Cost of noticing late: about
   * $12. A mode is a promise to do something specific instead of everything.
   */
  const modes = { recompute: RECOMPUTE, rejudge: REJUDGE, plan: PLAN } as Record<string, unknown>;
  const handled = new Set(["recompute", "rejudge", "plan"]);
  for (const [name, value] of Object.entries(modes)) {
    if (value && !handled.has(name)) {
      throw new Error(`הדגל --${name} הועבר אך אין ענף שמטפל בו — עצירה לפני ריצה מלאה בטעות`);
    }
  }

  if (REJUDGE) {
    const f = JSON.parse(readFileSync(REJUDGE, "utf8")) as {
      gold: GoldRow[]; controls: ControlRow[]; counters: typeof counters;
      metrics: Record<string, unknown>; valid: boolean;
      validity?: { complete: boolean; stagesRan: boolean; noFailures: boolean; noJudgeFailures: boolean };
    };
    const goldById = new Map(loadGold(GOLD_FILE).map(r => [r.id, r]));
    const broken = f.gold.filter(r => r.verdict?.judgeFailed);
    console.log(`${broken.length} שורות עם כישלון שופט`);

    let fixed = 0;
    for (const row of broken) {
      const rec = goldById.get(row.id);
      if (!rec) { console.log(`   ${row.id} לא נמצא בסט הזהב`); continue; }
      // A minimal Answer carrying exactly what the judge reads.
      const ans = {
        text: row.answerText,
        sources: row.sources.map(code => ({ code, title: "", score: 0 })),
        unverified: row.verdict.hasUnverified ? ["?"] : [],
      } as unknown as Answer;
      const v = await judgeAnswer(rec, ans);
      if (v.judgeFailed) { console.log(`   ${row.id} נכשל שוב`); continue; }
      row.verdict = v;
      row.costUSD += priceOf(v.usage.inputTokens, v.usage.outputTokens);
      fixed++;
      console.log(`   ${row.id} נשפט: עובדה=${v.factFound} (${v.factMethod}) · ציטוט=${v.expectCited} · נכונה=${v.correct}`);
    }

    const scorable = f.gold.filter(r => r.verdict.factMethod !== "skipped" && !r.verdict.judgeFailed);
    const correct = scorable.filter(r => r.verdict.correct).length;
    f.counters.judgeFailed = f.gold.filter(r => r.verdict?.judgeFailed).length;
    f.metrics.answerAccuracy = scorable.length ? correct / scorable.length : null;
    f.metrics.answerAccuracyN = `${correct}/${scorable.length}`;

    /**
     * Re-judging repairs ONE of the four validity conditions. The other three
     * were decided while the run was happening and cannot be re-derived from
     * the file, so they are read back from what the run recorded. A file with
     * no `validity` block predates that record and cannot be declared valid
     * here at all - conservative on purpose, because the alternative is a run
     * that failed a gate being promoted by a judge repair.
     */
    const recorded = f.validity;
    if (!recorded) {
      f.valid = false;
      console.log("\nהקובץ נוצר לפני שנרשמו תנאי התוקף, ולכן אי אפשר להכריז אותו תקף מכאן.");
    } else {
      recorded.noJudgeFailures = f.counters.judgeFailed === 0;
      f.valid = recorded.complete && recorded.stagesRan && recorded.noFailures && recorded.noJudgeFailures;
      const failing = Object.entries(recorded).filter(([, ok]) => !ok).map(([k]) => k);
      if (failing.length) console.log(`\nתנאים שעדיין נכשלים: ${failing.join(", ")}`);
    }
    writeFileSync(REJUDGE, JSON.stringify(f, null, 1), "utf8");
    console.log(`\nתוקנו ${fixed} · דיוק תשובה: ${correct}/${scorable.length} (${((correct / scorable.length) * 100).toFixed(1)}%) · תוקף: ${f.valid ? "תקפה" : "פסולה"}`);
    return;
  }

  if (RECOMPUTE) {
    const f = JSON.parse(readFileSync(RECOMPUTE, "utf8")) as { gold: GoldRow[]; controls: ControlRow[]; metrics: Record<string, unknown> };
    let changed = 0;
    for (const c of f.controls) {
      const before = c.refused;
      c.refused = isRefusal(c.answerText);
      c.refusedLoose = mentionsRefusal(c.answerText);
      if (before !== c.refused) changed++;
    }
    for (const g of f.gold) {
      const before = g.verdict.refused;
      g.verdict.refused = isRefusal(g.answerText);
      if (before !== g.verdict.refused) changed++;
    }
    const refused = f.controls.filter(c => c.refused).length;
    const loose = f.controls.filter(c => c.refusedLoose).length;
    f.metrics.refusals = `${refused}/${f.controls.length}`;
    f.metrics.refusalsLooseBaselineMethod = `${loose}/${f.controls.length}`;
    (f as Record<string, unknown>).recomputedAt = new Date().toISOString();
    writeFileSync(RECOMPUTE, JSON.stringify(f, null, 1), "utf8");
    console.log(`חושב מחדש ${RECOMPUTE}: ${changed} שורות שינו סיווג סירוב`);
    console.log(`סירובים: ${refused}/${f.controls.length} (בשיטת הבסיס הישנה: ${loose})`);
    return;
  }

  // Iron rule 1, enforced rather than remembered.
  if (existsSync(LOCK_PATH)) {
    const held = readFileSync(LOCK_PATH, "utf8");
    console.error(`מדידה אחרת כבר רצה (${held.trim()}).`);
    console.error(`מדידת Bedrock אחת בכל רגע — שתי ריצות מקבילות מייצרות 429 שקט ומכנה קטן.`);
    console.error(`אם זו ריצה שנתקעה: מחק ${LOCK_PATH} והרץ שוב.`);
    process.exit(2);
  }
  writeFileSync(LOCK_PATH, `pid ${process.pid} · ${new Date().toISOString()}\n`, "utf8");

  const t0 = Date.now();
  try {
    console.log(describeConfig());
    console.log(`אוסף: ${DEFAULT_COLLECTION} (${collectionDef(DEFAULT_COLLECTION).members?.length ?? 1} חברים) · מקביליות: ${CONCURRENCY}`);

    let gold = GOLD_ONLY || !CONTROLS_ONLY ? loadGold(GOLD_FILE) : [];
    let controls = CONTROLS_ONLY || !GOLD_ONLY ? loadControls(CONTROLS_FILE) : [];
    if (LIMIT) { gold = gold.slice(0, LIMIT); controls = controls.slice(0, LIMIT); }

    // Rows carried over from an earlier, incomplete run.
    let priorGold: GoldRow[] = [];
    let priorControls: ControlRow[] = [];
    if (RESUME) {
      const prior = JSON.parse(readFileSync(RESUME, "utf8")) as {
        gold: GoldRow[]; controls: ControlRow[]; config: string; counters: typeof counters;
      };
      // A row whose JUDGE failed carries no verdict worth keeping — ask it again.
      priorGold = prior.gold.filter(r => !r.verdict?.judgeFailed);
      priorControls = prior.controls;
      const haveGold = new Set(priorGold.map(r => r.id));
      const haveControls = new Set(priorControls.map(r => r.id));
      const beforeG = gold.length, beforeC = controls.length;
      gold = gold.filter(r => !haveGold.has(r.id));
      controls = controls.filter(r => !haveControls.has(r.id));
      console.log(`המשך ריצה מ-${RESUME}`);
      console.log(`   נשמרו ${priorGold.length}/${beforeG} זהב ו-${priorControls.length}/${beforeC} בקרות; נותרו לשאול ${gold.length} + ${controls.length}`);
      if (prior.config !== describeConfig()) {
        console.warn(`   *** אזהרה: הקונפיג שונה מזה של הריצה הקודמת. מיזוג שתי תצורות אינו מדידה אחת. ***`);
      } else {
        console.log(`   הקונפיג זהה לריצה הקודמת — המיזוג מודד תצורה אחת`);
      }
    }
    /**
     * A gold file may not be measured until its facts have been graded.
     *
     * This is a hard stop, not a warning, and it exists because the same mistake
     * happened twice: `--annotate` was run on gold.json and never on the annex
     * set, so a third of that set was scored by literal matching against values
     * like "20%" (183 occurrences in its own document) and "ירושלים" (40). The
     * run completed, reported a 12.5-point drop, and the number meant nothing.
     *
     * Annotation is cheap, local and needs no model call, so there is no reason
     * to ever measure without it — and no reason to leave the decision to
     * whoever remembers. The check runs BEFORE the lock and before a single
     * question is asked, so the cost of forgetting is a message, not $12.
     */
    const unannotated = gold.filter(r => r.fact && r.factOccurrences === undefined);
    if (unannotated.length) {
      console.error(`\n*** ${unannotated.length} מתוך ${gold.length} רשומות זהב ללא דירוג כוח הבחנה. ***`);
      console.error(`מדידה על סט לא מדורג אינה תקפה: ערך כמו "20%" שמופיע 183 פעם במסמכו מתאים לכל תשובה.`);
      console.error(`הרץ קודם (מקומי, ללא עלות):`);
      console.error(`   npx tsx src/eval/audit-gold.ts --annotate --gold=${GOLD_FILE ?? "eval/gold/gold.json"}`);
      process.exit(2);
    }

    console.log(`סט זהב: ${gold.length} · בקרות: ${controls.length}`);

    if (PLAN) {
      console.log(`
תוכנית: יישאלו ${gold.length} שאלות זהב ו-${controls.length} בקרות.`);
      console.log(`מוערך: ${(((gold.length + controls.length) * 0.0295)).toFixed(2)}$ · כ-${Math.round(((gold.length + controls.length) * 2.2) / 10)} דק' בקצב שנמדד (10 קריאות לדקה)`);
      console.log(`10 הראשונות: ${gold.slice(0, 10).map(g => g.id).join(", ")}`);
      return;
    }

    // Build the BM25 index before the clock starts, so its ~2s does not land on
    // the first question's latency and skew the p50.
    const members = collectionDef(DEFAULT_COLLECTION).members;
    if (members) { console.log("מחמם את מנוע המילים..."); await warmUnion(members); }

    const goldRows = [...priorGold, ...(gold.length ? await runGold(gold) : [])];
    const controlRows = [...priorControls, ...(controls.length ? await runControls(controls) : [])];

    // ------------------------------------------------------------- metrics
    const scorable = goldRows.filter(r => r.verdict.factMethod !== "skipped" && !r.verdict.judgeFailed);
    const correct = scorable.filter(r => r.verdict.correct).length;
    const skipped = goldRows.length - scorable.length;

    const recall = goldRows.filter(r => r.recallAt12).length;
    const top1 = goldRows.filter(r => r.top1).length;
    const top3 = goldRows.filter(r => r.top3).length;
    const refused = controlRows.filter(r => r.refused).length;
    const refusedLoose = controlRows.filter(r => r.refusedLoose).length;

    /**
     * MRR — the metric top-1 and top-3 cannot express.
     *
     * Both of those are binary, so "the instruction came second" and "the
     * instruction came eighth" score identically: zero. The whole point of the
     * ranking work in accuracy-plan stage 1-2 is to move the right instruction
     * UP, and a change that lifts it from 8th to 3rd is real progress that
     * top-1 reports as no progress at all. Reciprocal rank scores each position
     * as 1/rank, so that movement is visible.
     *
     * Derived entirely from rankedCodes, which every row already carries — no
     * extra model call, and it can be recomputed over old result files to get a
     * retroactive baseline (see scripts note in accuracy-plan §4 stage 0).
     * A miss scores 0, which is the standard convention and keeps the metric on
     * the same denominator as top-1.
     */
    const reciprocalRanks = goldRows.map(r => {
      const i = r.rankedCodes.indexOf(r.expect);
      return i >= 0 ? 1 / (i + 1) : 0;
    });
    const mrr = reciprocalRanks.length
      ? reciprocalRanks.reduce((s, x) => s + x, 0) / reciprocalRanks.length
      : 0;

    const latencies = goldRows.map(r => r.firstTokenMs).filter((x): x is number => x !== null);
    /**
     * The same rows with their backoffs taken out - what the system spends
     * WORKING before the first word. `latencies` above keeps its old
     * definition, unchanged, so this run stays comparable with every result
     * file written before 19/09/2026; this is a second number beside it, not a
     * new ruler. The gate is still read off `latencies`.
     */
    const latenciesWork = goldRows
      .filter(r => r.firstTokenMs !== null)
      .map(r => Math.max(0, (r.firstTokenMs as number) - (r.firstTokenWaitMs ?? 0)));
    /**
     * Work with BOTH kinds of contention removed: the backoffs and the round
     * trips of calls that were rejected. `latenciesWork` above keeps the
     * meaning it was first measured with on 19/09 (backoffs only), so the two
     * runs of that day stay comparable; this is the honest one to read against
     * the 6-second target.
     */
    const latenciesNet = goldRows
      .filter(r => r.firstTokenMs !== null)
      .map(r => Math.max(0, (r.firstTokenMs as number) - (r.firstTokenWaitMs ?? 0) - (r.firstTokenRejectedMs ?? 0)));
    const rowsWaitedBeforeFirstToken = goldRows.filter(r => (r.firstTokenWaitMs ?? 0) > 0).length;
    /** Where the waiting happened, across every answer in the run. */
    const waitByStage: { [stage: string]: { waitMs: number; retries: number } } = {};
    for (const r of [...goldRows, ...controlRows]) {
      for (const [stage, v] of Object.entries(r.retryByStage ?? {})) {
        const acc = (waitByStage[stage] ??= { waitMs: 0, retries: 0 });
        acc.waitMs += v.waitMs;
        acc.retries += v.retries;
      }
    }
    const cost = goldRows.reduce((s, r) => s + r.costUSD, 0) + controlRows.reduce((s, r) => s + r.costUSD, 0);
    const questions = goldRows.length + controlRows.length;

    const pct = (a: number, b: number): string => (b ? `${((a / b) * 100).toFixed(1)}%` : "—");

    /**
     * Valid means COMPLETE, not merely "nothing threw".
     *
     * The first full run had 110 failures and reported metrics over whatever
     * survived; the denominator, not the numerator, was the thing that moved.
     * So validity is defined against the sets on disk: every gold record and
     * every control must carry a verdict, and no judge call may have failed.
     */
    const expectedGold = (GOLD_ONLY || !CONTROLS_ONLY ? loadGold(GOLD_FILE).length : 0);
    const expectedControls = (CONTROLS_ONLY || !GOLD_ONLY ? loadControls(CONTROLS_FILE).length : 0);
    const complete = LIMIT
      ? counters.failed === 0
      : goldRows.length === expectedGold && controlRows.length === expectedControls;

    /**
     * A stage that was configured ON and did not run invalidates the run.
     *
     * Added 10/09/2026, and it is the SECOND half of the reranker fix. The
     * first half made the skips visible — counters.reranked and rerankSkipped
     * are printed and stored. But visible is not the same as blocking: the
     * 08/09 baseline would still have been stamped "תקפה" under the old rule,
     * because nothing threw and every question got an answer. It reported 76.3%
     * top-1 for a system whose ranking stage was absent in 226 of 300
     * questions, and that number went into CLAUDE.md, into the plan, and into
     * two days of decisions built on top of it.
     *
     * The lesson is not "watch the counter". A counter nobody is forced to read
     * is a counter that will not be read — the reranked field existed on every
     * row the whole time. So the run now refuses to call itself valid, the same
     * way it already refuses when the denominator shrank.
     *
     * Scoped to RERANK being on, because with RERANK=0 not running is correct.
     */
    // Rows the confidence gate refused never reach the reranker, by design, and
    // are excluded here — they still count as wrong in answer accuracy. Read off
    // the rows, so a resumed row from before the field existed (undefined)
    // stays in the denominator and cannot be excused.
    const gateRefusedGold = goldRows.filter(r => r.refusedAtGate === true).length;
    const rerankExpected = RERANK ? goldRows.length - gateRefusedGold : 0;
    /**
     * The same rule for REFUSAL_QUOTE_CHECK (14/09/2026): a verdict that did not
     * come back, or a refusal that shows sources without an "answers" verdict,
     * is a stage that did not run where it was configured to.
     */
    const quoteCheckRan = !REFUSAL_QUOTE_CHECK || (counters.quoteCheckFailed === 0 && counters.refusalSourcesUnchecked === 0);
    /** And for SOURCES_ADD_QUOTED_DOC (16/09/2026): every model-written answer carries its evidence. */
    const quotedDocsRan = counters.quotedDocsRan === counters.quotedDocsExpected;
    /** And for ANSWER_SCOPE_FIRST (22/09/2026): every model-written answer was sent the rule. */
    const scopeFirstRan = counters.scopeFirstRan === counters.scopeFirstExpected;
    /**
     * And for JUDGE_QUESTION_POLAR_KIND (17/09/2026): every kind-"כן/לא" row the
     * model judged must say the question-aware prompt scored it. Computed from the
     * rows, not a live counter, so resumed rows are checked too - and a resumed
     * row from before the evidence field existed cannot prove which judge scored
     * it, so it fails this check on purpose.
     */
    const modelJudged = goldRows.filter(r => r.verdict.factMethod === "model" && !r.verdict.judgeFailed);
    const judgedWithQuestion = modelJudged.filter(r => r.verdict.judgePrompt === "question").length;
    const judgeQuestionMissedIds = JUDGE_QUESTION_POLAR_KIND
      ? modelJudged.filter(r => r.kind === "כן/לא" && r.verdict.judgePrompt !== "question").map(r => r.id)
      : [];
    const stagesRan = counters.reranked >= rerankExpected && quoteCheckRan && quotedDocsRan && scopeFirstRan && judgeQuestionMissedIds.length === 0;

    const valid = complete && stagesRan && counters.failed === 0 && counters.judgeFailed === 0;

    console.log("\n" + "=".repeat(66));
    console.log(`מדד                          ערך`);
    console.log("-".repeat(66));
    console.log(`דיוק תשובה                   ${correct}/${scorable.length}  ${pct(correct, scorable.length)}`);
    console.log(`recall@12                    ${recall}/${goldRows.length}  ${pct(recall, goldRows.length)}`);
    console.log(`top-1 (אחרי דירוג)           ${top1}/${goldRows.length}  ${pct(top1, goldRows.length)}`);
    console.log(`top-3 (אחרי דירוג)           ${top3}/${goldRows.length}  ${pct(top3, goldRows.length)}`);
    console.log(`MRR (ממוצע 1/דירוג)          ${mrr.toFixed(3)}`);
    console.log(`סירובים בבקרות               ${refused}/${controlRows.length}  ${pct(refused, controlRows.length)}` +
      (refusedLoose !== refused ? `   (בשיטת הבסיס של 07/09: ${refusedLoose})` : ""));
    console.log(`טוקן ראשון p50 / p90         ${(percentile(latencies, 0.5) / 1000).toFixed(1)} / ${(percentile(latencies, 0.9) / 1000).toFixed(1)} שנ'`);
    /**
     * The same p50/p90 with contention peeled off in two steps, printed under
     * the gate line rather than instead of it. Each line says exactly what was
     * subtracted, so no reader has to guess which "latency" is meant: the first
     * removes the deliberate backoffs, the second also removes the round trips
     * of calls that were rejected and returned nothing.
     */
    console.log(
      `פחות המתנה בנסיגות          ${(percentile(latenciesWork, 0.5) / 1000).toFixed(1)} / ${(percentile(latenciesWork, 0.9) / 1000).toFixed(1)} שנ'`,
    );
    console.log(
      `פחות גם קריאות שנדחו        ${(percentile(latenciesNet, 0.5) / 1000).toFixed(1)} / ${(percentile(latenciesNet, 0.9) / 1000).toFixed(1)} שנ'` +
      `   (${rowsWaitedBeforeFirstToken}/${latencies.length} שורות נחנקו לפני המילה הראשונה)`,
    );
    console.log(`עלות                         $${cost.toFixed(3)} סה"כ · ${(questions ? (cost / questions) * 100 : 0).toFixed(2)} סנט לשאלה`);
    console.log("-".repeat(66));
    console.log(`נוסו ${counters.attempted} · נענו ${counters.answered} · נכשלו ${counters.failed} · throttle ${counters.throttled} · הבהרה ${counters.clarified} · שופט נפל ${counters.judgeFailed}`);
    /**
     * Iron rule 2 in its own line: before any wait number is read, say how many
     * answers carried a ledger at all. `retryMeterRows` below the answer count
     * means the meter did not run and the zeros are silence, not evidence.
     */
    console.log(
      `מד ההמתנה: רץ ב-${counters.retryMeterRows}/${counters.answered} תשובות · המתינו ${counters.retryRowsWaited} · ` +
      `סה"כ ${(counters.retryWaitTotalMs / 1000).toFixed(1)} שנ' ב-${counters.retryCountTotal} נסיגות` +
      ` · ועוד ${(counters.retryRejectedTotalMs / 1000).toFixed(1)} שנ' ב-${counters.retryRejectedCalls} קריאות שנדחו` +
      (Object.keys(waitByStage).length
        ? ` · ${Object.entries(waitByStage).map(([k, v]) => `${k} ${v.retries}×${(v.waitMs / 1000).toFixed(1)}שנ'`).join(", ")}`
        : "") +
      (counters.retryMeterRows === counters.answered ? "" : "   *** המד לא רץ בכל התשובות ***"),
    );
    /**
     * Proof that the region rotation ran, printed whenever GEN_REGIONS is set.
     *
     * The failure this guards against has already happened once on this
     * project, on the lexical side: a metric read the same before and after a
     * change, and the reading was filed as "it does not help" for days before
     * it turned out the component had never executed. A rotation is especially
     * easy to get wrong that way - it changes no text, so the only visible
     * symptom of a rotation that quietly collapsed onto one region is a latency
     * number that did not move.
     *
     * A run whose rotation reached fewer regions than were configured says so
     * in the line rather than in a file nobody opens.
     */
    const regionCalls: Record<string, number> = {};
    for (const r of [...goldRows, ...controlRows]) {
      for (const [region, n] of Object.entries(r.genRegions ?? {})) {
        regionCalls[region] = (regionCalls[region] ?? 0) + n;
      }
    }
    if (GEN_REGIONS.length) {
      const reached = Object.keys(regionCalls).length;
      console.log(
        `רוטציית regions: ${GEN_REGIONS.length} מוגדרים · הגיעו ל-${reached} · ` +
        Object.entries(regionCalls).sort((a, b) => b[1] - a[1]).map(([r, n]) => `${r} ${n}`).join(", ") +
        (reached === GEN_REGIONS.length ? "" : "   *** הרוטציה לא הגיעה לכל ה-regions ***"),
      );
    }
    console.log(
      `דירוג רץ ב-${counters.reranked}/${goldRows.length} שאלות זהב` +
      (gateRefusedGold ? ` · ${gateRefusedGold} נעצרו בשער הביטחון לפני הדירוג (בכוונה, נספרות כשגויות)` : "") +
      (Object.keys(rerankSkipped).length
        ? ` · לא רץ: ${Object.entries(rerankSkipped).map(([k, v]) => `${k}=${v}`).join(", ")}`
        : "") +
      (counters.reranked >= rerankExpected ? "" : "   *** זו לבדה פוסלת את הריצה ***"),
    );
    if (UNVERIFIED_ALLOWS_PASSAGE_REFS) {
      console.log(`הפניות מתוך הקטע: הכלל רץ על ${counters.passageRefsRan} תשובות · הסיר אזהרה ב-${counters.passageRefsCleared}`);
    }
    if (SOURCES_ADD_QUOTED_DOC) {
      console.log(
        `מקור מתוך ציטוט: הכלל רץ על ${counters.quotedDocsRan}/${counters.quotedDocsExpected} תשובות · הוסיף מסמך ב-${counters.quotedDocsUsed}` +
        (quotedDocsRan ? "" : "   *** זו לבדה פוסלת את הריצה ***"),
      );
    }
    if (ANSWER_SCOPE_FIRST) {
      console.log(
        `תחולה לפני תשובה: החוק נשלח ב-${counters.scopeFirstRan}/${counters.scopeFirstExpected} תשובות שנכתבו במודל` +
        (scopeFirstRan ? "" : "   *** זו לבדה פוסלת את הריצה ***"),
      );
    }
    if (JUDGE_WITH_QUESTION || JUDGE_QUESTION_POLAR_KIND) {
      console.log(
        `שופט עם שאלה: ${judgedWithQuestion}/${modelJudged.length} שורות שנשפטו במודל` +
        (judgeQuestionMissedIds.length ? `   *** בלי שאלה למרות הדגל: ${judgeQuestionMissedIds.join(", ")} — זו לבדה פוסלת את הריצה ***` : ""),
      );
    }
    if (PARENT_TOP_DOC_SLACK) {
      console.log(`מרווח למסמך הראשון (${PARENT_TOP_DOC_SLACK}): הכלל רץ על ${counters.topDocSlackRan} תשובות · הכניס מסמך שלם ב-${counters.topDocSlackUsed}`);
    }
    if (REFUSAL_QUOTE_CHECK) {
      console.log(
        `בדיקת ציטוט בסירוב: עונה ${counters.quoteCheckAnswers} · מסביר ${counters.quoteCheckExplains} · נכשלה ${counters.quoteCheckFailed}` +
        ` · סירוב עם מקורות בלי אישור ${counters.refusalSourcesUnchecked}` +
        (quoteCheckRan ? "" : "   *** זו לבדה פוסלת את הריצה ***"),
      );
    }
    console.log(
      `דגל סירוב מבני תואם לבדיקת הטקסט ב-${counters.answered - counters.refusedFlagMismatch}/${counters.answered} תשובות` +
      (refusedFlagMismatchIds.length ? `   *** חריגות: ${refusedFlagMismatchIds.join(", ")} ***` : ""),
    );
    console.log(`רשומות ללא fact מאומת (מחוץ למכנה): ${skipped}`);
    console.log(`שלמות: ${goldRows.length}/${expectedGold} זהב · ${controlRows.length}/${expectedControls} בקרות`);
    console.log(`תוקף הריצה: ${valid ? "תקפה" : "*** פסולה ***"}`);
    console.log("=".repeat(66));

    if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const outPath = `${RESULTS_DIR}/${stamp}.json`;
    writeFileSync(outPath, JSON.stringify({
      valid,
      /**
       * The four conditions behind `valid`, recorded rather than left to be
       * recomputed later.
       *
       * --rejudge used to rebuild `valid` from two of them (no failures, no
       * judge failures) and silently drop the other two. On the 10/09 gate run
       * that would have flipped a file to valid while the reranker had skipped
       * 59 of 300 questions - laundering exactly the defect the gate was added
       * to catch, two days after it was added. A verdict that can be rebuilt
       * from a subset of its evidence eventually will be.
       */
      validity: {
        complete,
        stagesRan,
        noFailures: counters.failed === 0,
        noJudgeFailures: counters.judgeFailed === 0,
      },
      at: new Date().toISOString(),
      resumedFrom: RESUME ?? null,
      minutes: Number(((Date.now() - t0) / 60000).toFixed(1)),
      config: describeConfig(),
      collection: DEFAULT_COLLECTION,
      concurrency: CONCURRENCY,
      counters,
      refusedFlagMismatchIds,
      /** Judge-prompt evidence (17/09/2026). See judgeQuestionMissedIds above. */
      judgeQuestion: { modelJudged: modelJudged.length, withQuestion: judgedWithQuestion, missedIds: judgeQuestionMissedIds },
      rerankSkipped,
      failureDetail,
      metrics: {
        answerAccuracy: scorable.length ? correct / scorable.length : null,
        answerAccuracyN: `${correct}/${scorable.length}`,
        recallAt12: goldRows.length ? recall / goldRows.length : null,
        top1: goldRows.length ? top1 / goldRows.length : null,
        top3: goldRows.length ? top3 / goldRows.length : null,
        mrr: goldRows.length ? Number(mrr.toFixed(4)) : null,
        refusals: `${refused}/${controlRows.length}`,
        refusalsLooseBaselineMethod: `${refusedLoose}/${controlRows.length}`,
        firstTokenP50Ms: percentile(latencies, 0.5),
        firstTokenP90Ms: percentile(latencies, 0.9),
        /**
         * Added 19/09/2026. The two lines above keep the old definition (they
         * include every backoff); these say how much of that was waiting for
         * the account's rate quota, so the 6-second target can be read against
         * work rather than against contention.
         */
        firstTokenWorkP50Ms: percentile(latenciesWork, 0.5),
        firstTokenWorkP90Ms: percentile(latenciesWork, 0.9),
        firstTokenNetP50Ms: percentile(latenciesNet, 0.5),
        firstTokenNetP90Ms: percentile(latenciesNet, 0.9),
        retryRejectedTotalMs: counters.retryRejectedTotalMs,
        retryRejectedCalls: counters.retryRejectedCalls,
        rowsWaitedBeforeFirstToken,
        retryMeterRows: counters.retryMeterRows,
        retryRowsWaited: counters.retryRowsWaited,
        retryWaitTotalMs: counters.retryWaitTotalMs,
        retryWaitByStage: waitByStage,
        costUSD: Number(cost.toFixed(4)),
        centsPerQuestion: questions ? Number(((cost / questions) * 100).toFixed(3)) : null,
        skippedNoFact: skipped,
      },
      gold: goldRows,
      controls: controlRows,
    }, null, 1), "utf8");
    console.log(`\nתוצאות: ${outPath}`);

    if (!valid) {
      console.error(`\nהריצה פסולה: ${counters.failed} כישלונות, ${counters.judgeFailed} כישלונות שופט.`);
      if (counters.throttled) console.error(`יש throttle — הרץ שוב עם EVAL_CONCURRENCY=1.`);
      if (!complete) console.error(`חסרות תשובות: ${expectedGold - goldRows.length} זהב, ${expectedControls - controlRows.length} בקרות. הרץ שוב עם --resume=${outPath}`);
      if (failureDetail.length) console.error(JSON.stringify(failureDetail.slice(0, 10), null, 1));
      process.exitCode = 1;
    }
  } finally {
    if (existsSync(LOCK_PATH)) unlinkSync(LOCK_PATH);
  }
}

main().catch(e => { console.error(e); if (existsSync(LOCK_PATH)) unlinkSync(LOCK_PATH); process.exit(1); });
