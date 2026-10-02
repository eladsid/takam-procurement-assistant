/**
 * judge.ts — decide whether one answer was CORRECT, not merely well-retrieved.
 *
 * THE THREE QUESTIONS, AND WHY ALL THREE ARE NEEDED
 *   (a) Did the expected fact reach the reader?   -> the answer is useful
 *   (b) Was the expected instruction cited?       -> the answer is checkable
 *   (c) Were there invented citations?            -> the answer is trustworthy
 * Correct means all three. Dropping (b) would credit an answer that happens to
 * contain "7 ימים" while citing the wrong regulation — the most dangerous
 * failure this system can produce, because it reads as authoritative and a
 * reader who knows the rules catches it immediately. Dropping (c) would credit
 * an answer that carried the right fact under a made-up instruction number.
 *
 * WHY LITERAL FIRST, MODEL SECOND
 * A literal match is deterministic, free and instant, and after the generator
 * was tightened most facts are short values ("7 ימים", "0.5%", "חשב המשרד")
 * that an answer reproduces verbatim. The model is the fallback for the honest
 * paraphrase — "שבעה ימים", "חצי אחוז" — and it is asked one closed question
 * with a one-word answer, because a judge that is allowed to explain is a judge
 * that is allowed to rationalise.
 *
 * WHY A FAILED JUDGE CALL IS NOT "INCORRECT"
 * A throttled judge would silently lower the accuracy score, and a measurement
 * that degrades under load in the direction of looking worse is exactly the
 * trap iron rule 7 exists for. A judge failure is recorded as its own outcome
 * and invalidates the run.
 */

import { containsNormalised, isRefusal, REFUSAL_PREFIX, REFUSAL_LEAD as REFUSAL_LEAD_RE, type GoldRecord } from "./gold.js";
import type { Answer } from "../core/types.js";
import { JUDGE_WITH_QUESTION, JUDGE_QUESTION_POLAR_KIND } from "../core/config.js";
import { readFileSync, readdirSync } from "node:fs";

export interface Verdict {
  /** Did the expected fact appear in the answer text? */
  factFound: boolean;
  /** How that was decided. "skipped" = the record has no verified fact to check. */
  factMethod: "literal" | "model" | "skipped";
  /** Was the expected instruction among the cited sources? */
  expectCited: boolean;
  /** Did the answer cite instruction numbers that match no retrieved passage? */
  hasUnverified: boolean;
  /** factFound && expectCited && !hasUnverified. False whenever the fact was skipped. */
  correct: boolean;
  /** The answer was the fixed refusal sentence. A refusal to a gold question is a miss. */
  refused: boolean;
  /** The judge's own model call failed. Not a verdict — a hole in the measurement. */
  judgeFailed?: boolean;
  /**
   * Which prompt the model judge was handed. Present only when factMethod is
   * "model". The evidence field for JUDGE_WITH_QUESTION / JUDGE_QUESTION_POLAR_KIND
   * (17/09/2026): a verdict that did not move under a judge flag means nothing
   * until this says the flag's prompt was the one that ran.
   */
  judgePrompt?: "plain" | "question";
  /**
   * Tokens the judge itself spent on this verdict.
   *
   * Carried on the verdict rather than in a module-level counter: the runner
   * judges two questions at a time, and a shared mutable "last usage" would be
   * overwritten by whichever call finished second. The cost line would then be
   * quietly too low — the same class of defect as the eval script that counted
   * 34 requests as 50.
   */
  usage: { inputTokens: number; outputTokens: number };
}

const SYSTEM_JUDGE = `אתה בודק תשובות. תקבל עובדה ותשובה.
ענה במילה אחת בלבד: כן — אם העובדה נמסרה בתשובה, גם אם בניסוח אחר או במילים אחרות. לא — אם היא לא נמסרה.
"כן" רק אם הערך עצמו נמסר. ערך שונה, ערך קרוב או היעדר הערך = לא.
אל תסביר. אל תוסיף מילה. פלט תקין: כן`;

/**
 * The closed question for a YES/NO fact, asked with the question it answers
 * (JUDGE_WITH_QUESTION). Used for no other fact - see YES_NO_FACT below.
 *
 * The strictness lines of the original are kept word for word, and no clause
 * was added that forgives hedging or exceptions: every sentence of leniency in
 * a judge prompt is one a wrong answer can hide behind.
 */
const SYSTEM_JUDGE_WITH_QUESTION = `אתה בודק תשובות. תקבל שאלה, את הערך הנכון שעונה עליה, ותשובה שנכתבה לשאלה.
ענה במילה אחת בלבד: כן — אם התשובה שנכתבה מוסרת לשאלה את הערך הנכון, גם אם בניסוח אחר או במילים אחרות. לא — אם היא לא מוסרת אותו.
ערך קצר כמו "לא" או "כן" הוא התשובה לשאלה עצמה: "כן" רק אם התשובה שנכתבה עונה לשאלה באותו כיוון.
"כן" רק אם הערך עצמו נמסר. ערך שונה, ערך קרוב או היעדר הערך = לא.
אל תסביר. אל תוסיף מילה. פלט תקין: כן`;

/**
 * Which facts get the question: exactly "כן" or "לא", and nothing else.
 *
 * Measured 13/09/2026 on the model-judged rows of two runs, the SAME stored
 * answers scored by both judges, plus an adversarial pass that pairs each
 * question with another question's answer:
 *
 *   yes/no facts (6 rows gate run, 5 rows 08/09 baseline)
 *     old judge: rejects g-154 g-182 g-213, all three an explicit and cited
 *                "לא"; ACCEPTS g-099 and g-103 with each other's "כן" answer
 *     question:  rescues all three, zero right answers lost, zero adversarial
 *                accepts
 *   every other fact (45 + 40 rows)
 *     question, first wording:  +1 / -4 - g-291 answers "כמה פעמים בשנה" with
 *                "4 פעמים בשנה" against "מדי 3 חודשים" and was rejected
 *     question, softer wording: +1 / -2 on the gate run, +3 / -1 on the
 *                baseline, and it began accepting adversarial answers
 *                (g-055, g-005) - and on yes/no it accepted g-182 with g-150's
 *                unrelated "אינה מכירה"
 *
 * "60%" or "חשב המשרד" is readable without its question; asking the judge to
 * weigh the question as well only gives it a second thing to disagree with.
 * "לא" is not readable without it. So the question goes exactly where the fact
 * cannot be read alone.
 *
 * The reverse direction - does it REJECT right answers? - measured 15/09/2026
 * (eval/benches/2026-09-15-yes-no-judge-flip). The trap feared was the one the
 * refusal check fell into on 14/09: a one-word "כן/לא" verdict about a yes/no
 * question comes back as an answer to the question, not as a classification.
 * Its signature needs no ground truth: a judge that answers the question gives
 * the same verdict whatever value it is handed. So every stored answer to the
 * six yes/no records (18 distinct texts, all runs) was judged with the true
 * value AND the flipped one, twice:
 *
 *   old judge (no question)  true value accepted 6/18, and 1/13 on "לא" facts;
 *                            same verdict for both values on 11/18 - it does
 *                            not classify "לא" at all; g-154 of the gate run
 *                            accepted "כן" for "לא יחול באופן רטרואקטיבי"
 *   this prompt              true 32/36, flipped 0/36, repeats identical,
 *                            adversarial pairs 0/30
 *   same prompt, labels      identical to this prompt on all 36 and 0/30 - so
 *   תואם/שונה                the labels stay: the explicit value to compare
 *                            against is what keeps it a classification
 *
 * The four rejections are both g-150 texts rejected for BOTH values: one is a
 * refusal that never answers (right), the other is the gate run's "אינה מכירה,
 * למעט במקרים מוגבלים" (a strict reading of "ערך קרוב = לא", not the trap).
 *
 * Widened to gold kind "כן/לא" behind JUDGE_QUESTION_POLAR_KIND, measured
 * 17/09/2026 (eval/benches/2026-09-17-judge-polar-kind). The target: g-196,
 * "לא. בן משפחה ... אינו רשאי לנהוג" against "אינם רשאים", rejected in all four
 * runs that wrote it. The scope reaches 9 distinct stored texts across 40
 * result files (g-108 g-132 g-161 g-167 g-196); read by hand, 7 deliver the
 * value and 2 (g-132) do not. Same three tests as above, same prompt:
 *
 *   old judge (no question)  true value 6/7 right, 0/2 wrong; g-196 rejected for
 *                            the true AND the flipped value ("רשאים") - it does
 *                            not classify a polar phrase either
 *   this prompt              true value 7/7 right, 0/2 wrong, both repeats
 *                            identical; flipped value rejected on every text
 *                            but one (see below)
 *   adversarial              each widened question against every other kind
 *                            "כן/לא" answer of the 4-flag gate run: 148 pairs on
 *                            the model path; old judge accepted 5 (three of
 *                            them for g-196 - "לא יהיו זכאים" read as "אינם
 *                            רשאים"), this prompt 0
 *
 * The one flipped value both judges accepted is a bench artefact, not the trap:
 * one g-167 text describes BOTH configurations, so "בתצורה זו יש יכולת לאתר"
 * is also in it. On the production judge, stored answers of both valid gate
 * runs: exactly g-196 moves (to correct), 297/300 rows untouched by scope.
 */
const YES_NO_FACT = /^(כן|לא)$/;

const JUDGE_TIMEOUT_MS = 30_000;

interface JudgeCall { verdict: boolean | null; usage: { inputTokens: number; outputTokens: number } }

/**
 * One closed-question model call, retried, with the failure surfaced rather than swallowed.
 * `question` present = the question-aware prompt; absent = the original prompt, unchanged.
 */
async function modelSaysFactPresent(fact: string, answerText: string, question?: string): Promise<JudgeCall> {
  const { getLlmTransport } = await import("../core/answer.js");
  const spent = { inputTokens: 0, outputTokens: 0 };
  const [system, user] = question === undefined
    ? [SYSTEM_JUDGE, `העובדה: ${fact}\n\nהתשובה:\n${answerText}`]
    : [SYSTEM_JUDGE_WITH_QUESTION, `השאלה: ${question}\n\nהערך הנכון: ${fact}\n\nהתשובה שנכתבה:\n${answerText}`];
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await Promise.race([
        getLlmTransport().generate(system, user),
        new Promise<never>((_, rej) => setTimeout(() => rej(new Error("judge timeout")), JUDGE_TIMEOUT_MS)),
      ]);
      const t = String(res.text ?? "").trim();
      spent.inputTokens += res.inputTokens ?? 0;
      spent.outputTokens += res.outputTokens ?? 0;
      if (/^כן/.test(t)) return { verdict: true, usage: spent };
      if (/^לא/.test(t)) return { verdict: false, usage: spent };
      // Anything else is not a verdict. Retry rather than guess at it.
    } catch {
      // fall through to the retry
    }
    await new Promise(r => setTimeout(r, 800 * attempt));
  }
  return { verdict: null, usage: spent };
}

/**
 * Instruction code -> full text, built once from the indexes on disk.
 *
 * The judge needs the corpus for exactly one question — "does the instruction
 * this answer cited actually contain the fact?" — and it is the eval harness,
 * so reading the index files is fair game. Lazy because a run that never
 * reaches this path should not pay for it.
 */
let corpusByCode: Map<string, string> | null = null;
function instructionText(code: string): string {
  if (!corpusByCode) {
    // node:fs is imported at the top: a require() here broke under any caller
    // with top-level await (ERR_AMBIGUOUS_MODULE_SYNTAX, 13/09/2026) in this
    // "type": "module" package. The READ stays lazy; only the import moved.
    corpusByCode = new Map();
    for (const f of readdirSync("data").filter(f => /^takam-\d+\.json$/.test(f))) {
      const o = JSON.parse(readFileSync("data/" + f, "utf8")) as { chunks?: { code: string; text?: string }[] };
      for (const c of o.chunks ?? []) {
        corpusByCode.set(c.code, (corpusByCode.get(c.code) ?? "") + "\n" + (c.text ?? ""));
      }
    }
  }
  return corpusByCode.get(code) ?? "";
}

/**
 * Is the answer CHECKABLE — not "did it name the one instruction we wrote down".
 *
 * The old test was `sources.includes(rec.expect)`, and it was too strict in a
 * way that cost real points. Measured on the 10/09 gate run: of the ten answers
 * that carried the right fact but were scored as wrong-citation, **seven cited
 * an instruction whose own text contains that same fact.** Chapter 14.2 is the
 * clearest case — three separate gold questions expect 14.2.10, and the model
 * cited 14.2.8 and 14.2.9, which say the same thing. TAKAM repeats a rule
 * across sibling instructions; the gold record names one of them, and naming
 * one made the other siblings wrong by fiat.
 *
 * The safety property this test exists for is untouched. The dangerous failure
 * — the right fact under a regulation that does not support it — is still
 * scored wrong, because the cited instruction must CONTAIN the fact. What is no
 * longer scored wrong is citing a different instruction that does support it.
 *
 * Note this changes the definition of the metric: `expectCited` on runs before
 * 11/09/2026 was measured under the stricter rule and is not comparable on this
 * axis. A `weakFact` record is excluded from the widening — "לא" appears in
 * every Hebrew paragraph, so a substring test there would accept anything.
 */
function citationSupportsFact(rec: GoldRecord, ans: Answer): boolean {
  if (ans.sources.some(s => s.code === rec.expect)) return true;
  if (!rec.fact || rec.weakFact) return false;
  return ans.sources.some(s => containsNormalised(instructionText(s.code), rec.fact));
}

/**
 * Does this record's model verdict get the question-aware prompt?
 *
 * JUDGE_WITH_QUESTION: only a fact that is exactly "כן"/"לא".
 * JUDGE_QUESTION_POLAR_KIND: also every record whose gold kind is "כן/לא" - the
 * polar verb phrases ("אינם רשאים", "רשות") that are as unreadable without their
 * question as a bare "לא". The bare facts stay in scope under either flag.
 */
function questionInScope(rec: GoldRecord, fact: string, withQuestion: boolean, polarKind: boolean): boolean {
  const bare = YES_NO_FACT.test(fact.trim());
  return (withQuestion && bare) || (polarKind && (bare || rec.kind === "כן/לא"));
}

/**
 * `opts.withQuestion` / `opts.polarKind` override the two judge flags for one
 * call. They exist so different judges can score the SAME stored answers in one
 * process - the only comparison in which a change of verdict is the judge's
 * doing and not the generator's. The runner never passes them; the flags decide.
 *
 * A caller that pins `withQuestion` and says nothing about `polarKind` gets
 * polarKind=false, not the environment's value: the benches of 13/09 and 15/09
 * pass `withQuestion:false` to mean "the old judge", and a flag set in the shell
 * must not quietly turn that into the new one.
 */
export async function judgeAnswer(
  rec: GoldRecord,
  ans: Answer,
  opts: { withQuestion?: boolean; polarKind?: boolean } = {},
): Promise<Verdict> {
  const noCost = { inputTokens: 0, outputTokens: 0 };
  const withQuestion = opts.withQuestion ?? JUDGE_WITH_QUESTION;
  const polarKind = opts.polarKind ?? (opts.withQuestion === undefined ? JUDGE_QUESTION_POLAR_KIND : false);

  const refused = isRefusal(ans.text);
  const expectCited = citationSupportsFact(rec, ans);
  const hasUnverified = ans.unverified.length > 0;

  // A record whose fact could not be verified against the source (the imported
  // 76 that failed enrichment) cannot be scored for answer accuracy. It still
  // contributes to recall and ranking; the runner keeps it out of the accuracy
  // denominator and reports how many were excluded.
  if (!rec.fact) {
    return { factFound: false, factMethod: "skipped", expectCited, hasUnverified, correct: false, refused, usage: noCost };
  }

  /**
   * A refusal that goes on to answer anyway is judged on what it went on to say.
   *
   * The old rule here was "a refusal cannot contain the fact", and it was wrong
   * as a matter of fact rather than of taste. Measured on the 10/09 gate run:
   * FIVE of the six rows scored as refusals contain the expected value in their
   * text. The model opens with the refusal sentence and then answers — and the
   * sharpest case, g-027, is one where the refusal IS the answer: the question
   * is whether a pilot may be extended, the correct value is "לא", and
   * "לא מצאתי במסמכים שברשותי" was scored as a miss for saying so.
   *
   * So: strip the refusal sentence and look at what remains. Nothing left means
   * a genuine refusal, and a refusal to a gold question is still a miss. Text
   * left means the model answered, and that text is judged like any other.
   *
   * `refused` stays true either way — it is an observation about the answer's
   * shape, and the control set relies on it. What changes is only whether that
   * observation is allowed to short-circuit the verdict.
   */
  const remainder = ans.text.trim().replace(REFUSAL_LEAD_RE, "").slice(REFUSAL_PREFIX.length).replace(/^[.\s]+/, "");
  if (refused && !remainder.trim()) {
    return { factFound: false, factMethod: "literal", expectCited, hasUnverified, correct: false, refused, usage: noCost };
  }
  /** What the fact is looked for in: the answer, minus a refusal preamble if there was one. */
  const judged = refused ? remainder : ans.text;

  /**
   * The literal shortcut is skipped for a fact with no discriminative power
   * (see GoldRecord.weakFact). For those the model is the only honest judge:
   * a substring search for "לא" succeeds against any Hebrew paragraph.
   */
  if (!rec.weakFact && containsNormalised(judged, rec.fact)) {
    const correct = expectCited && !hasUnverified;
    return { factFound: true, factMethod: "literal", expectCited, hasUnverified, correct, refused, usage: noCost };
  }

  const askWithQuestion = questionInScope(rec, rec.fact, withQuestion, polarKind);
  const judgePrompt = askWithQuestion ? "question" : "plain";
  const { verdict, usage } = await modelSaysFactPresent(rec.fact, judged, askWithQuestion ? rec.q : undefined);
  if (verdict === null) {
    return {
      factFound: false, factMethod: "model", expectCited, hasUnverified,
      correct: false, refused, judgeFailed: true, judgePrompt, usage,
    };
  }

  return {
    factFound: verdict,
    factMethod: "model",
    expectCited,
    hasUnverified,
    correct: verdict && expectCited && !hasUnverified,
    refused,
    judgePrompt,
    usage,
  };
}
