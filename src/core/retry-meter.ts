/**
 * Retry meter — tells "working" apart from "waiting for capacity", per answer.
 *
 * WHY THIS EXISTS (18/09/2026). The latency target is "first token within 6
 * seconds", and the gate run of 15/09 recorded p90 = 9.1s. Nothing had been
 * added that writes more text between the 5.6s baseline and that number; what
 * had been added to the clock was WAITING. The same run recorded 503 throttles
 * across 400 questions, and every backoff inside `ask()` — the query embedding,
 * the reranker, the answer call — sleeps while the stopwatch in run-eval keeps
 * running. Measured on 30 of those questions one at a time: p90 8.3s including
 * throttled rows, but 4.2s on the 15 rows whose first attempt was clean.
 *
 * So the single number `firstTokenMs` answers two different questions at once,
 * and we cannot tell whether the system is slow or the account's rate quota is
 * full. This module makes them two numbers:
 *
 *     firstTokenMs = firstTokenWorkMs + firstTokenWaitMs
 *
 * WHAT IT IS NOT. It changes no behaviour whatsoever: the same calls are made,
 * the same backoffs are slept, the same answers come back. It only records how
 * long a sleep took and which stage slept — which is why it is not behind a
 * config flag. A flag guards a component that can change an answer; nothing
 * here can. The gate metric `firstTokenMs` keeps its old definition, and the
 * decomposition is reported ALONGSIDE it rather than replacing it, so every
 * result file written before today stays comparable.
 *
 * WHY AsyncLocalStorage AND NOT A MODULE-LEVEL COUNTER. `run-eval` asks several
 * questions at once (EVAL_CONCURRENCY), and a shared counter would charge one
 * question's wait to whichever answer happened to finish next — a number that
 * looks perfectly plausible and is wrong. A store scoped to one `ask()` call
 * follows that call across every await inside it and nowhere else. Outside a
 * scope (ingest, a bench, a bare script) every function here is a no-op, so
 * nothing needs to know whether it is being metered.
 */
import { AsyncLocalStorage } from "node:async_hooks";

/**
 * The stages that can sleep inside one `ask()`. Kept as a closed union rather
 * than a free string so a new backoff site has to declare where it lives; a
 * ledger that says only "3.4 seconds of waiting" cannot answer the question
 * that follows it, which is always "waiting for which model".
 */
export type RetryStage = "embed" | "rerank" | "answer" | "quote-check";

export interface RetryLedger {
  /** Total measured time slept in backoffs, in ms. */
  waitMs: number;
  /** How many backoffs were slept (not how many calls were made). */
  retries: number;
  /**
   * Time spent inside calls that were REJECTED and produced nothing - measured
   * from sending the request to the throttle coming back.
   *
   * Added 19/09/2026, after the first metered run showed why it is needed: the
   * 17 rows that were never throttled had a p90 of 5.10s, while "work" on the
   * throttled rows came out at 8.64s. The gap is not the system being slower
   * on those questions; it is the round trip of a call that was turned away.
   * Charging it to work says the assistant is slow, which is false; charging
   * it to the quota says the account is full, which is what happened.
   *
   * Kept apart from waitMs rather than folded into it, so the numbers measured
   * on 19/09 keep the meaning they were measured with.
   */
  rejectedMs: number;
  rejectedCalls: number;
  /** The same numbers per stage; a stage that never stalled is absent. */
  byStage: { [K in RetryStage]?: { waitMs: number; retries: number; rejectedMs: number; rejectedCalls: number } };
  /** Error names that caused the waits, first seen first, de-duplicated. */
  reasons: string[];
  /**
   * How many generation calls this answer sent to each region, added 20/09/2026
   * for GEN_REGIONS.
   *
   * This is the evidence field, not a metric. The rule it serves: before
   * reading whether a number moved, prove the component ran. A rotation that
   * silently resolved to one region would leave the latency unchanged and look
   * exactly like "spreading the load does not help" — the same mistake already
   * paid for once, when BM25 read 45% before and after because it was never
   * running. With this field a results file states, per row, which ceilings
   * actually served it.
   *
   * Empty when the rotation is off, which is also a true statement about the
   * run: nothing rotated.
   */
  regions: Record<string, number>;
}

const store = new AsyncLocalStorage<RetryLedger>();

export const newRetryLedger = (): RetryLedger =>
  ({ waitMs: 0, retries: 0, rejectedMs: 0, rejectedCalls: 0, byStage: {}, reasons: [], regions: {} });

/** The per-stage record, created on first use so an untouched stage stays absent. */
const stageOf = (led: RetryLedger, stage: RetryStage) =>
  (led.byStage[stage] ??= { waitMs: 0, retries: 0, rejectedMs: 0, rejectedCalls: 0 });

/** Run `fn` with `ledger` as the active scope. Nested scopes shadow, not merge. */
export function runWithRetryLedger<T>(ledger: RetryLedger, fn: () => Promise<T>): Promise<T> {
  return store.run(ledger, fn);
}

/**
 * Record one backoff. `waitMs` is the MEASURED elapsed time around the sleep,
 * not the number the caller asked to sleep: a 500ms timer on a busy event loop
 * is not 500ms, and the point of this file is to account for real seconds.
 */
export function recordRetryWait(stage: RetryStage, waitMs: number, reason?: string): void {
  const led = store.getStore();
  if (!led) return;                       // not metered — ingest, bench, CLI
  led.waitMs += waitMs;
  led.retries += 1;
  const s = stageOf(led, stage);
  s.waitMs += waitMs;
  s.retries += 1;
  if (reason && !led.reasons.includes(reason)) led.reasons.push(reason);
}

/**
 * Record one call that was sent and turned away. `elapsedMs` is the round trip
 * of that failed call: time the caller spent with nothing to show for it.
 */
export function recordRejectedCall(stage: RetryStage, elapsedMs: number, reason?: string): void {
  const led = store.getStore();
  if (!led) return;
  led.rejectedMs += elapsedMs;
  led.rejectedCalls += 1;
  const s = stageOf(led, stage);
  s.rejectedMs += elapsedMs;
  s.rejectedCalls += 1;
  if (reason && !led.reasons.includes(reason)) led.reasons.push(reason);
}

/**
 * Record that one generation call was addressed to `region`. Counted whether
 * the call succeeded or was thrown back: "which ceilings did this answer ask"
 * is the question, and a refusal is an ask.
 */
export function recordRegionCall(region: string): void {
  const led = store.getStore();
  if (!led) return;
  led.regions[region] = (led.regions[region] ?? 0) + 1;
}

/**
 * A snapshot of the ledger as it stands right now, or undefined outside a
 * scope. It is a copy on purpose: the caller that reads it at first-token time
 * keeps a number from that instant, and the live ledger goes on growing while
 * the rest of the answer is written.
 */
export function readRetryLedger(): RetryLedger | undefined {
  const led = store.getStore();
  if (!led) return undefined;
  const byStage: RetryLedger["byStage"] = {};
  for (const [k, v] of Object.entries(led.byStage)) byStage[k as RetryStage] = { ...v! };
  return {
    waitMs: led.waitMs,
    retries: led.retries,
    rejectedMs: led.rejectedMs,
    rejectedCalls: led.rejectedCalls,
    byStage,
    reasons: [...led.reasons],
    regions: { ...led.regions },
  };
}

/** One-line form for a results file or a log: "answer 2×1512ms · rerank 1×501ms". */
export function describeRetryLedger(led: RetryLedger): string {
  const parts = Object.entries(led.byStage).map(
    ([k, v]) => `${k} ${v!.retries}×${v!.waitMs}ms + ${v!.rejectedCalls} rejected ${v!.rejectedMs}ms`,
  );
  return parts.length ? parts.join(" · ") : "none";
}
