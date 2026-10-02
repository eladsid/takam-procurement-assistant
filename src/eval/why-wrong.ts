/**
 * why-wrong.ts — where in the chain did each wrong answer break?
 *
 * WHY THIS EXISTS
 * A run reports one number for answer accuracy and four for retrieval, and the
 * plan then argues about which stage to work on next. That argument was being
 * settled by intuition. On 10/09/2026 it was settled by this breakdown instead,
 * and the answer overturned the queue: of 24 wrong answers in the gate run, 4
 * were retrieval (the instruction was never retrieved, or the ranker dropped
 * it) and 20 were the answer stage — a refusal opener on top of an answer that
 * contains the fact, a right fact attributed to a neighbouring instruction, or
 * a value read wrong from the right document. Perfect retrieval was worth 1.5
 * points; the next two items in the queue were both retrieval items.
 *
 * WHAT IT COSTS
 * Nothing. It reads a results file that is already on disk and makes no model
 * call. That is the point: a diagnosis that costs a run is a diagnosis nobody
 * repeats, and this one should run after every run.
 *
 * HOW A ROW IS ATTRIBUTED
 * To the FIRST stage that lost it, walking the pipeline in order. A row that
 * was never retrieved is not also blamed on the answer prompt. The denominator
 * is exactly run-eval's `scorable` — records with no checkable value are out of
 * the accuracy metric, so they are out of its breakdown too, or the parts would
 * not sum to the whole.
 *
 * WHAT IT DOES NOT DO
 * It does not re-judge and it does not overrule the judge. A row it puts in
 * `right-doc-wrong-value` may still be a judging error rather than an answering
 * one — see the weakFact note printed at the end.
 *
 * USAGE
 *   npm run eval:why-wrong                        # newest file in eval/results
 *   npm run eval:why-wrong -- <path> --list       # and print every failing row
 */

import { readFileSync, readdirSync } from "node:fs";
import { loadGold, containsNormalised, type GoldRecord } from "./gold.js";

const RESULTS_DIR = "eval/results";

interface Verdict {
  factFound: boolean;
  factMethod: string;
  expectCited: boolean;
  correct: boolean;
  refused: boolean;
  judgeFailed?: boolean;
}

interface GoldRow {
  id: string;
  q: string;
  expect: string;
  fact: string | null;
  retrievedCodes: string[];
  rankedCodes: string[];
  reranked?: boolean;
  sources: string[];
  answerText: string;
  verdict: Verdict;
}

interface ResultsFile {
  valid: boolean;
  at: string;
  config?: string;
  counters?: { reranked?: number };
  metrics?: Record<string, unknown>;
  gold: GoldRow[];
}

/** In pipeline order. The first one that is true owns the row. */
const STAGES = [
  ["search-ceiling", "ההוראה לא הייתה בין המועמדים בכלל"],
  ["ranker-dropped", "אוחזרה, והדירוג הוציא אותה מהרשימה"],
  ["refused-though-ranked", "ההוראה דורגה, והתשובה נפתחה בסירוב"],
  ["answered-cited-other", "ענה, אך ייחס את התשובה להוראה אחרת"],
  ["right-doc-wrong-value", "ההוראה הנכונה, הערך שגוי"],
] as const;

type Stage = (typeof STAGES)[number][0];

function stageOf(row: GoldRow): Stage | "ok" {
  if (row.verdict.correct) return "ok";
  if (!row.retrievedCodes.includes(row.expect)) return "search-ceiling";
  if (!row.rankedCodes.includes(row.expect)) return "ranker-dropped";
  // Checked before the citation test on purpose: a refusal is stripped of its
  // sources by answer.ts, so a refused row would otherwise always look like a
  // citation failure and the two would be impossible to tell apart.
  if (row.verdict.refused) return "refused-though-ranked";
  if (!row.verdict.expectCited) return "answered-cited-other";
  return "right-doc-wrong-value";
}

function newestResults(): string {
  const files = readdirSync(RESULTS_DIR)
    .filter(f => /^\d{4}-\d{2}-\d{2}T/.test(f) && f.endsWith(".json"))
    .sort();
  const last = files[files.length - 1];
  if (!last) throw new Error(`אין קובצי תוצאות ב-${RESULTS_DIR}`);
  return `${RESULTS_DIR}/${last}`;
}

function main(): void {
  const args = process.argv.slice(2);
  const list = args.includes("--list");
  const path = args.find(a => !a.startsWith("--")) ?? newestResults();

  const f = JSON.parse(readFileSync(path, "utf8")) as ResultsFile;
  const goldById = new Map<string, GoldRecord>(loadGold().map(r => [r.id, r]));

  // The same filter run-eval uses for the accuracy denominator.
  const scorable = f.gold.filter(r => r.verdict.factMethod !== "skipped" && !r.verdict.judgeFailed);
  const buckets = new Map<Stage | "ok", GoldRow[]>();
  for (const row of scorable) {
    const s = stageOf(row);
    if (!buckets.has(s)) buckets.set(s, []);
    buckets.get(s)!.push(row);
  }

  const pct = (n: number): string => `${((n / scorable.length) * 100).toFixed(1)}%`;
  const ok = buckets.get("ok")?.length ?? 0;

  console.log("=".repeat(72));
  console.log(`${path}`);
  console.log(`תוקף: ${f.valid ? "תקפה" : "*** פסולה ***"} · ${f.at} · דירוג רץ ב-${f.counters?.reranked ?? "?"} שאלות`);
  if (f.config) console.log(f.config);
  console.log("-".repeat(72));
  console.log(`נכונות                       ${String(ok).padStart(3)}/${scorable.length}  ${pct(ok)}`);
  console.log(`שגויות                       ${String(scorable.length - ok).padStart(3)}/${scorable.length}  ${pct(scorable.length - ok)}`);
  console.log("-".repeat(72));

  let retrievalSide = 0;
  let answerSide = 0;
  for (const [stage, hebrew] of STAGES) {
    const n = buckets.get(stage)?.length ?? 0;
    if (stage === "search-ceiling" || stage === "ranker-dropped") retrievalSide += n;
    else answerSide += n;
    console.log(`${stage.padEnd(24)} ${String(n).padStart(3)}  ${pct(n).padStart(6)}   ${hebrew}`);
  }
  console.log("-".repeat(72));
  console.log(`צד האחזור: ${retrievalSide} (${pct(retrievalSide)})   ·   צד התשובה: ${answerSide} (${pct(answerSide)})`);
  console.log(`תקרת הרווח מאחזור מושלם: ${pct(retrievalSide)} — כל השאר נמצא אחרי שההוראה הנכונה כבר הייתה ביד.`);

  /**
   * Two checks on the judge, printed with the breakdown rather than separately,
   * because both inflate the failure count without any system defect:
   *
   *   1. judge.ts short-circuits a refusal to factFound=false, on the stated
   *      assumption that "a refusal cannot contain the fact". It can: the model
   *      opens with the refusal sentence and then quotes the answer.
   *   2. a weakFact record ("לא") is judged by a model, and a model asked
   *      whether "לא" appears in a Hebrew paragraph is not a reliable adjudicator.
   */
  const refusalsHoldingFact: string[] = [];
  for (const row of buckets.get("refused-though-ranked") ?? []) {
    const rec = goldById.get(row.id);
    if (!rec?.fact || rec.weakFact) continue;
    if (containsNormalised(row.answerText ?? "", rec.fact)) refusalsHoldingFact.push(row.id);
  }
  const weakWrong = scorable.filter(r => !r.verdict.correct && goldById.get(r.id)?.weakFact).map(r => r.id);

  console.log("-".repeat(72));
  console.log(`סירובים שהטקסט שלהם מכיל את העובדה: ${refusalsHoldingFact.length}` +
    (refusalsHoldingFact.length ? `  (${refusalsHoldingFact.join(", ")})` : "") +
    "  — נספרו כשגויות, והתשובה בהן קיימת");
  console.log(`שגויות שערכן חסר כוח הבחנה (weakFact, נשפט במודל): ${weakWrong.length}` +
    (weakWrong.length ? `  (${weakWrong.join(", ")})` : ""));

  if (list) {
    for (const [stage] of STAGES) {
      for (const row of buckets.get(stage) ?? []) {
        console.log(`\n[${stage}] ${row.id} · expect=${row.expect} · דורג ראשון=${row.rankedCodes[0] ?? "—"} · ציטט=${row.verdict.expectCited}`);
        console.log(`   שאלה: ${row.q}`);
        console.log(`   ערך מצופה: ${row.fact}`);
        console.log(`   תשובה: ${(row.answerText ?? "").replace(/\s+/g, " ").slice(0, 240)}`);
      }
    }
  }
  console.log("=".repeat(72));
}

main();
