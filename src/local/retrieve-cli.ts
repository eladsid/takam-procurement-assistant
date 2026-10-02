/**
 * retrieve-cli.ts — inspect retrieval on its own, before the model is involved.
 *
 * This tool exists because of a rule that saves days: if the wrong passages
 * come back here, no amount of prompt engineering downstream will produce a
 * right answer. Verifying retrieval first turns "the answer is bad" from a
 * mystery into one of two specific, separately-fixable problems.
 *
 * It runs the SAME query preparation the real answer path runs — including the
 * translation of a Hebrew question against an English corpus — because a
 * measuring tool that measures something other than production is worse than
 * no measuring tool: it produces numbers you then trust.
 *
 * Usage:  npm run retrieve -- "מתי מותר להתקשר בפטור ממכרז?"
 *         npm run retrieve -- --collection=aws-all "מה זה cold start"
 *         npm run retrieve -- --raw --collection=aws-lambda "מה זה cold start"
 *         PARENT_VOTE=1 npm run retrieve -- --k=12 "השאלה"   (רשימה לפני ואחרי הצבעת הורה)
 */

import { retrieve, retrieveAcross, isConfident, parentVote } from "../core/retrieve.js";
import { prepareQuery } from "../core/query-prep.js";
import { resolveMembers } from "../core/store.js";
import { DEFAULT_COLLECTION, collectionDef, minScoreFor, describeConfig, PARENT_VOTE } from "../core/config.js";

const args = process.argv.slice(2);
const collection = args.find(a => a.startsWith("--collection="))?.split("=")[1] ?? DEFAULT_COLLECTION;
// --raw skips query preparation, which is how the before/after of the
// translation step is measured rather than asserted.
const raw = args.includes("--raw");
// How wide a list to show. Default stays TOP_K so the tool's existing output is
// unchanged; --k=12 reproduces the candidate list the reranker actually sees,
// which is the only list on which parent vote means anything.
const k = Number(args.find(a => a.startsWith("--k="))?.split("=")[1]) || undefined;
const question = args.filter(a => !a.startsWith("--")).join(" ");

if (!question) {
  console.error('שימוש: npm run retrieve -- "השאלה שלך"');
  process.exit(1);
}

const def = collectionDef(collection);
const members = await resolveMembers(def);

console.log(`\n${describeConfig()}`);
console.log(`\nשאלה: ${question}`);
console.log(`אוסף:  ${collection}${members.length ? `  (${members.length} אוספים מוטמעים מתוך ${def.members!.length} מוצהרים)` : ""}`);

const prepared = raw
  ? { queries: [question], translation: undefined, note: "‎--raw: ללא הכנת שאילתה" }
  : await prepareQuery(question, collection);

if (prepared.translation) console.log(`נוסח נוסף שנחפש בו: ${prepared.translation}`);
if (prepared.note) console.log(`הערה: ${prepared.note}`);
console.log();

const hits = members.length
  ? await retrieveAcross(prepared.queries, members, k)
  : await retrieve(prepared.queries, collection, k);

hits.forEach((h, i) => {
  const from = members.length ? `  [${collectionDef(h.collection).label}]` : "";
  console.log(`[${i + 1}] ${h.score.toFixed(3)}  ${h.code} — ${h.title}${from}`);
  if (h.keywords?.length) console.log(`    מילות מפתח: ${h.keywords.slice(0, 6).join(", ")}`);
  console.log(`    ${h.text.slice(0, 200).replace(/\s+/g, " ")}...\n`);
});

/**
 * Parent vote, before and after — the evidence that the stage RAN.
 *
 * Rule 2 of the accuracy plan: a metric that did not move means "the component
 * did not run" until proven otherwise, and BM25 was silently dead three times
 * for exactly that reason. So the tool prints the list the reranker would have
 * received without the vote and the list it receives with it, side by side, and
 * says which documents were promoted and on how many chunks.
 */
if (PARENT_VOTE) {
  const pv = await parentVote(hits);
  console.log(`
— הצבעת הורה: ${pv.applied ? "פעלה" : "לא שינתה דבר"} · קודמו ${pv.promoted.length} הוראות —`);
  for (const p of pv.promoted) {
    console.log(`    ${p.code}: ${p.votes} קטעים ברשימה מתוך ${p.total} · היה במקום ${p.wasAt}`);
  }
  const before = hits.map(h => h.code).join(" > ");
  const after = pv.hits.map(h => h.code).join(" > ");
  console.log(`  לפני: ${before}`);
  console.log(`  אחרי: ${after}`);
  console.log(`  ${hits.length} מועמדים → ${pv.hits.length}
`);
}

// The refusal threshold is shown explicitly so it can be calibrated by eye
// against a question that IS covered and one that deliberately is not.
console.log(
  isConfident(hits, collection)
    ? `✓ מעל הסף (${minScoreFor(collection)}) — המערכת תענה`
    : `✗ מתחת לסף (${minScoreFor(collection)}) — המערכת תשיב "לא מצאתי במסמכים שברשותי."`,
);
