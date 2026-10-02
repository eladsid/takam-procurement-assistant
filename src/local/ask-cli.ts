/**
 * ask-cli.ts — ask one question from the terminal, end to end.
 *
 * Usage:  npm run ask -- "מתי מותר להתקשר בפטור ממכרז ומי הגורם המאשר?"
 *         npm run ask -- --collection=aws-bedrock "what is a guardrail"
 */

import { ask } from "../core/answer.js";
import { DEFAULT_COLLECTION, describeConfig } from "../core/config.js";

const args = process.argv.slice(2);
const collection = args.find(a => a.startsWith("--collection="))?.split("=")[1] ?? DEFAULT_COLLECTION;
const question = args.filter(a => !a.startsWith("--")).join(" ");

if (!question) {
  console.error('שימוש: npm run ask -- "השאלה שלך"');
  process.exit(1);
}

console.log(`\n${describeConfig()}\n`);

const answer = await ask(question, collection);

console.log(`${answer.text}\n`);

if (answer.truncated) {
  console.log("⚠ התשובה נקטעה: המודל הגיע לתקרת אורך הפלט (GEN_MAX_TOKENS).\n");
  process.exitCode = 2;
}

if (answer.sources.length) {
  console.log("מקורות שצוטטו:");
  for (const s of answer.sources) {
    console.log(`  • ${s.code} — ${s.title}  (קרבה ${s.score.toFixed(3)})`);
  }
} else {
  console.log("(אין מקורות מצוטטים)");
}

// The passages the system read but did not cite. Shown because "what did it
// look at" is as interesting as "what did it say" when judging a RAG system.
if (answer.retrieved.length) {
  console.log("\nנקרא ולא צוטט:");
  for (const r of answer.retrieved.filter(r => !answer.sources.some(s => s.code === r.code))) {
    console.log(`  · ${r.code} — ${r.title}  (${r.score.toFixed(3)})`);
  }
}

// An invented citation number is the error a reader who knows the regulations
// spots first, so it is reported loudly rather than left for them to find.
if (answer.unverified.length) {
  console.log(`\n⚠ ציטוטים לא מאומתים: ${answer.unverified.join(", ")}`);
  console.log("  המספרים האלה מופיעים בתשובה אך לא באף קטע שאותר — יש לאמת ידנית.");
  process.exitCode = 2;
}

console.log(`\nמודל: ${answer.model}  |  מסלול: ${answer.route}`);
console.log(`מאגר המסמכים נכון ל-${new Date(answer.indexedAt).toLocaleDateString("he-IL")}`);
console.log(
  `טוקנים: ${answer.usage.inputTokens} קלט / ${answer.usage.outputTokens} פלט` +
    `  |  עלות: $${answer.usage.costUSD.toFixed(5)}`,
);
