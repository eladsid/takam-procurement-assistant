/**
 * ingest-cli.ts — build a collection's index from the command line.
 *
 * A thin wrapper: it parses arguments, prints progress, and reports what the
 * run cost. All of the actual work lives in core/ingest.ts, which is the same
 * code the S3-triggered Lambda runs (AD-1).
 *
 * Usage:
 *   npm run ingest                        # the default collection (takam)
 *   npm run ingest -- --collection=takam
 *   npm run ingest -- --fresh             # ignore existing vectors, re-embed all
 */

import { ingest } from "../core/ingest.js";
import { DEFAULT_COLLECTION, collectionDef, describeConfig } from "../core/config.js";

const args = process.argv.slice(2);
const flag = (name: string): string | undefined =>
  args.find(a => a.startsWith(`--${name}=`))?.split("=").slice(1).join("=");

const collection = flag("collection") ?? DEFAULT_COLLECTION;

/**
 * A virtual collection has no documents of its own — it is a name for a search
 * across several real indexes. Since DEFAULT_COLLECTION became "takam-all", a
 * bare `npm run ingest` would otherwise try to build an index for something
 * that cannot have one, and fail with a confusing "no documents found".
 */
if (collectionDef(collection).members?.length) {
  console.error(
    `
"${collection}" הוא אוסף וירטואלי — הוא מאגד אוספים אחרים ואין לו מסמכים משלו.
` +
    `לבליעת כל פרקי התכ"ם:  npm run ingest:takam
` +
    `לבליעת אוסף בודד:      npm run ingest -- --collection=takam-7
`,
  );
  process.exit(1);
}
const fresh = args.includes("--fresh");

console.log(`\nבליעת אוסף "${collection}"`);
console.log(`  ${describeConfig()}\n`);

const result = await ingest({
  collection,
  reuseExisting: !fresh,
  onLog: line => console.log(line),
});

console.log(`\nהושלם ב-${(result.elapsedMs / 60000).toFixed(1)} דקות`);
console.log(`  מסמכים: ${result.documents}  |  קטעים: ${result.chunks}`);
console.log(`  הוטמעו עכשיו: ${result.embedded}  |  נעשה שימוש חוזר: ${result.reused}`);
console.log(`  טוקנים: ${result.tokens.toLocaleString()}  |  עלות: $${result.costUSD.toFixed(4)}`);

if (result.failures.length) {
  // Surfaced loudly rather than buried: a partially-ingested corpus that looks
  // complete is how a demo ends up unable to answer one specific question.
  console.log(`\n⚠ ${result.failures.length} מסמכים נכשלו:`);
  for (const f of result.failures) console.log(`  • ${f.key} — ${f.error}`);
  process.exitCode = 1;
}
