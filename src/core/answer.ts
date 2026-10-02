/**
 * answer.ts — the "question track", stage 2: turn retrieved passages into an answer.
 *
 * The single most important property of this file is what it does NOT allow:
 * the model may not use anything it knows about Israeli procurement law. It may
 * only use the passages handed to it, it must cite which instruction each fact
 * came from, and when the passages do not contain the answer it must say so.
 *
 * That constraint is the whole product. An assistant that guesses a threshold
 * in a procurement regulation is worse than no assistant at all, because a
 * confident wrong answer is acted upon.
 *
 * Two things changed relative to the first version of this logic (src/ask.ts),
 * both because a review found them wrong, and both marked "BUGFIX" below:
 *   1. the retry loop used to swallow EVERY error, so a malformed request was
 *      retried pointlessly and only the last, least informative error surfaced;
 *   2. `sources` used to list every retrieved passage while claiming in its own
 *      comment to list only the used ones — the citation box was lying.
 *
 * A third property is new: when retrieval is not confident, the model is never
 * called at all. The refusal is produced by code, not by a prompt.
 */

import { BedrockRuntimeClient, ConverseCommand, ConverseStreamCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  LLM_ROUTE,
  GEN_REGION,
  GEN_REGIONS,
  GEN_MODELS,
  GEN_MAX_TOKENS,  PARENT_CONTEXT,  PARENT_VOTE,  REFUSAL_OFFERS_TOPICS,
  ANSWER_STYLE,
  CONCISE_STYLE_RULES,
  ANSWER_APPLICABILITY,
  ANSWER_SCOPE_FIRST,
  REFUSAL_KEEPS_QUOTED_SOURCES,
  REFUSAL_QUOTE_CHECK,
  UNVERIFIED_ALLOWS_PASSAGE_REFS,
  SOURCES_ADD_QUOTED_DOC,
  OPENROUTER_API_KEY,
  ANTHROPIC_API_KEY,
  TOP_K,
  RERANK,
  RERANK_CANDIDATES,
  DEFAULT_COLLECTION,
  collectionDef,
} from "./config.js";
import { rerank } from "./rerank.js";
import type { Answer, GenerateResult, Hit, LlmTransport } from "./types.js";
import { retrieve, retrieveAcross, isConfident, withNeighbours, withParentDocuments, parentVote } from "./retrieve.js";
import { decide, buildClarification } from "./clarify.js";  // topics on a model-written refusal too
import { recordChoice } from "./feedback.js";
import { loadIndex, resolveMembers } from "./store.js";
import { prepareQuery } from "./query-prep.js";
import { newRetryLedger, runWithRetryLedger, recordRetryWait, recordRejectedCall, recordRegionCall, type RetryStage } from "./retry-meter.js";

// ---------------------------------------------------------------------------
// The prompt — the actual product
// ---------------------------------------------------------------------------

/**
 * The exact sentence the system must produce when the corpus does not answer
 * the question. It is a constant and not a literal in three places because two
 * other things depend on it being character-for-character identical: the
 * acceptance criterion for the "deliberately uncovered question" demo, and the
 * check below that keeps a refusal from being decorated with fake sources.
 */
export const REFUSAL_TEXT = "לא מצאתי במסמכים שברשותי.";

/**
 * The same sentence WITHOUT its full stop, and the test that actually works.
 *
 * REFUSAL_TEXT is exact only on the deterministic path, where the confidence
 * gate short-circuits and returns the constant. Since that gate became a cheap
 * floor (MIN_SCORE 0.49 -> 0.30) the usual refusal is written by the MODEL,
 * which continues the sentence and puts a markdown heading in front of it:
 *
 *     # תשובה
 *
 *     לא מצאתי במסמכים שברשותי הנחיות בדבר ...
 *
 * Measured 08/09/2026 over 100 adjacent control questions: matching the exact
 * constant recognised 87, this test recognises 94, and a bare `includes`
 * recognises 98 — the last being wrong in the dangerous direction, because it
 * counts answers that ANSWERED the out-of-scope question and only appended a
 * caveat (one was a complete how-to on opening a business bank account, with
 * the phrase at character 908 of 983).
 *
 * What the old test cost in production: a model-written refusal was treated as
 * an answer, so the `sources` fallback below attached the top passage to it and
 * the UI rendered "I did not find anything" next to a confident citation.
 */
export const REFUSAL_PREFIX = "לא מצאתי במסמכים שברשותי";

/**
 * Leading markdown the model puts in FRONT of a refusal, stripped before the
 * prefix test. Two forms, both measured in the wild:
 *   - a heading and a blank line, found 08/09/2026;
 *   - emphasis around the sentence itself (**...**), found 10/09/2026 in a
 *     control question about mortgages. The model DID refuse, correctly, and
 *     the run scored it as an answer -- a measurement error that reads as a
 *     safety regression, which is the expensive direction to be wrong in.
 *
 * This is the fourth time this one-line test has been wrong, always the same
 * way: the refusal's IDENTITY is being recovered from its rendering. The real
 * fix is a boolean decided once, where the refusal is produced -- see the
 * note in CLAUDE.md. Until then, this strips what is known to appear.
 */
export const REFUSAL_LEAD = /^(?:#{1,6}[^\n]*\n+|[*_]{1,3}|\s)+/;

export const isRefusalText = (text: string): boolean =>
  text.trim().replace(REFUSAL_LEAD, "").startsWith(REFUSAL_PREFIX);

/**
 * Rule 2 of the system prompt, in its two forms. See ANSWER_APPLICABILITY in
 * config.ts for the measurement that produced the second one.
 *
 * The shipped form (BASE) already allows a partial answer. What it does not do
 * is tell the model that a RULE settling the question counts as an answer to
 * the question, so a model holding "a framework tender shall be held as a
 * public tender only" and asked "may a framework tender be closed?" reads the
 * corpus as silent, refuses, and then answers anyway in the next sentence.
 */
const RULE2_BASE = `2. אם התשובה אינה נמצאת בקטעים — כתוב בדיוק: "${REFUSAL_TEXT}" ואל תוסיף ניחוש.
   אם הקטעים עונים על חלק מהשאלה בלבד — ענה על החלק שיש וציין במפורש מה לא נמצא. סירוב מלא שמור למצב שבו אין בקטעים כלום רלוונטי.`;

const RULE2_APPLICABILITY = `2. אם הקטעים מכילים כלל, הגדרה או איסור שממנו נובעת התשובה לשאלה — ענה מתוכו וציין את ההוראה, גם אם אין בקטעים אמירה מפורשת בניסוח של השאלה. כלל שמכריע את השאלה הוא תשובה לשאלה.
   אם הכלל חל על חלק מהמקרה בלבד, או שהוא חל על גוף אחר מזה שבשאלה — ענה ממנו וציין במפורש מה אינו חל ועל מי.
   אם הקטעים עונים על חלק מהשאלה בלבד — ענה על החלק שיש וציין במפורש מה לא נמצא.
   רק אם אין בקטעים שום בסיס לתשובה — כתוב בדיוק: "${REFUSAL_TEXT}" ואל תוסיף ניחוש. אל תפתח במשפט סירוב תשובה שאתה ממשיך ועונה בה.`;

/**
 * ANSWER_SCOPE_FIRST's rule (config.ts has the measurement). Appended after rule 8
 * on TAKAM collections only, and never replaces rule 2.
 *
 * Three wording choices carry the whole risk, so they are spelled out:
 *   • "ורק אחריו תבוא התשובה מתוכו" — the rule reorders an answer; it is not a
 *     refusal trigger. c-021 and c-098 become "these passages govern X; for X: ...",
 *     which is what the stage-4 exit gate already counts as correct.
 *   • The last sentence pins rule 2: WHEN to refuse is unchanged. Without it, "say
 *     whom it governs and then answer from it" reads as licence to answer c-071 /
 *     c-076 ("annual leave for a private-sector employee") from the contractors'
 *     annex — turning a correct refusal into a scoped answer, the expensive direction.
 *   • The examples are body TYPES that recur across the manual (suppliers and
 *     contractors, state loans, government housing), not the two failing questions'
 *     topics; "pension" and "mortgage" appear nowhere in the rule.
 * Exported so a bench can confirm the prompt that was sent actually carried it.
 */
export const SCOPE_FIRST_RULE = `9. תחולה לפני תשובה: אם הכלל שבקטעים חל על גוף או הסדר מסוים (למשל ספקים וקבלנים של המדינה, הלוואות מכספי המדינה, דיור ממשלתי), והשאלה נשאלת באופן כללי בלי לומר שהמקרה שלה שייך לאותו גוף או הסדר — המשפט הראשון בתשובה יאמר על מי הכלל חל, ורק אחריו תבוא התשובה מתוכו. במקרה כזה אל תפתח ב"כן", ב"לא" או במספר, ואל תנסח את הכלל כאילו הוא עוסק במקרה שבשאלה.
   חוק זה קובע רק איך נפתחת תשובה שאתה נותן. הוא אינו משנה מתי לסרב — זה נקבע בחוק 2.`;

/**
 * Quote matching, for the refusal rule below. Whitespace and markdown only.
 *
 * Deliberately NOT the evaluator's `containsNormalised` in eval/gold.ts, and the
 * difference is not an oversight. That one also folds quotes, niqqud and final
 * letters, because it compares a human-written expected VALUE against a model's
 * paraphrase. This one compares a span the model copied out of a passage against
 * that same passage, so the only gap to close is rendering — the model re-wraps
 * lines and adds `**` around the part it wants to stress. Folding more than that
 * would make the test easier to pass, and this test is the only thing standing
 * between a quoted citation and an invented one.
 */
const normaliseQuote = (t: string): string =>
  t.replace(/\*\*/g, "").replace(/\s+/g, " ").trim();

/**
 * The spans an answer PRESENTS as quotation: inside double quotes, or on a
 * markdown blockquote line. 25 characters minimum, because a short span matches
 * by coincidence — "המשרד" appears in nearly every instruction in the corpus.
 *
 * BUGFIX 16/09/2026 — a Hebrew abbreviation is not a quotation mark.
 * TAKAM writes its abbreviations with a plain double quote between two letters:
 * תכ"ם, מע"מ, ש"ח, בל"מ. The pairing regex counted that mark as a delimiter, so
 * one abbreviation shifted every pair after it by one: the function returned the
 * text BETWEEN quotations ("\n\n2. **חופשה מיוחדת** — (הוראה 14.2.8...") and
 * never saw the quotation itself. Measured on g-096 in the valid gate run: the
 * answer quotes 14.2.10 verbatim after a quote that contains 'תכ"ם', and the
 * sentence was invisible to every rule that reads quotes. 40 of 300 gate
 * answers extract a different span set once the mark is handled.
 *
 * The fix hides a letter-quote-letter mark behind a private-use character while
 * pairing and puts it back in the span, so the span still matches the passage
 * text (which keeps the plain quote). Blast radius on the refusal path,
 * derived on the 4-flag gate before this line changed: gold refusals 0 changed;
 * 3 control refusals keep a different set (c-016 more, c-017 none, c-061 one) and
 * all three get "מסביר" from checkRefusalQuote (c-061 checked 3/3), so zero
 * controls would show a source. eval/benches/2026-09-16-quote-attribution/.
 */
const ABBREVIATION_MARK = /(?<=[א-ת])"(?=[א-ת])/g;
const MARK_PLACEHOLDER = String.fromCharCode(0xe000);   // private use area: never in the corpus or in a model's answer

export function quotedSpans(text: string): string[] {
  const guarded = text.replace(ABBREVIATION_MARK, MARK_PLACEHOLDER);
  const out: string[] = [];
  for (const m of guarded.matchAll(/"([^"]{25,})"/g)) out.push(m[1]);
  for (const m of guarded.matchAll(/^\s*>\s*"?(.{25,})$/gm)) out.push(m[1]);
  return out
    .map(s => normaliseQuote(s.split(MARK_PLACEHOLDER).join('"')))
    .filter(s => s.length >= 25);
}

/**
 * SOURCES_ADD_QUOTED_DOC (config.ts has the measurement): the documents an
 * ANSWER quotes from that its own labels do not name.
 *
 * For each quoted span that no cited document contains, every document among
 * the passages the model was shown that does contain it is returned. Additive
 * by construction - the caller appends, never replaces - because the suppressing
 * direction was measured to touch 18 correct answers.
 *
 * Same containment test as quotedHits (the whole shown text of a document, not
 * the first hit citedHits keeps), for the reason documented there.
 */
export function quotedDocuments(text: string, cited: Hit[], passages: Hit[]): Hit[] {
  const spans = quotedSpans(text);
  if (spans.length === 0) return [];

  const key = (h: Hit) => `${h.collection}::${h.code}`;
  /**
   * One entry per document: its first passage (for title/url), its joined,
   * normalised text, and its best score - the same reason citedHits carries the
   * best score: the first passage of a document is often a score-0 neighbour.
   */
  const docs = new Map<string, { hit: Hit; body: string }>();
  for (const p of passages) {
    const d = docs.get(key(p));
    if (d) {
      d.body += "\n" + (p.text ?? "");
      if (p.score > d.hit.score) d.hit = { ...d.hit, score: p.score };
    } else {
      docs.set(key(p), { hit: { ...p }, body: p.text ?? "" });
    }
  }
  for (const d of docs.values()) d.body = normaliseQuote(d.body);

  const citedKeys = new Set(cited.map(key));
  const citedBodies = [...docs.entries()].filter(([k]) => citedKeys.has(k)).map(([, d]) => d.body);

  const out = new Map<string, Hit>();
  for (const s of spans) {
    if (citedBodies.some(b => b.includes(s))) continue;   // the label already points at the origin
    for (const [k, d] of docs) {
      if (!citedKeys.has(k) && d.body.includes(s)) out.set(k, d.hit);
    }
  }
  return [...out.values()];
}

/**
 * Of the passages this answer cites, the ones it actually QUOTES.
 *
 * This exists to separate two things that a refusal's text can be doing with an
 * instruction number, which the shipped code treats as one.
 *
 * A refusal that explains itself NAMES what it rejected: "I did not find it; the
 * passages I found concern VAT codes in Sigma (הודעה 2.2.3.1)". Putting 2.2.3.1
 * under "מקורות" — the heading that means "this is what the answer rests on" —
 * is a false claim, and removing it was the right call on 08/09/2026.
 *
 * A refusal that goes on to answer QUOTES what it used: g-190 writes
 * 'הוראה 7.3.5 קובעת בסעיף 1.2 כי "מכרז מסגרת ייערך אך ורק כמכרז פומבי"'. That
 * citation is load-bearing. Deleting it leaves the reader a complete, correct,
 * quoted answer beside an empty source box and no way to click through.
 *
 * Measured 11/09/2026 on the paired subset run, which is why the rule is a
 * quotation test and not a shape test. Three candidate discriminators were
 * available and two are useless: "is there text after the refusal sentence" is
 * true for 87 of 91 control refusals, and "does the answer name an instruction"
 * is true for almost all of them too. The quotation test splits the populations:
 * of 6 gold refusals it recovers exactly the 3 that carry the expected value,
 * and of 10 control refusals it leaves 9 with an empty box.
 *
 * It cannot invent a citation. The input is already `citedHits`, so the output
 * is a SUBSET of what the model itself wrote, filtered to what the corpus
 * confirms it copied. The `hits[0]` fallback stays switched off for refusals.
 *
 * `passages` is the third argument for a reason that cost a measurement to find.
 * The quote has to be looked for in EVERY passage of the cited document, not in
 * the one hit `citedHits` returned. citedHits emits one row per document, and
 * under PARENT_CONTEXT the model is handed the whole instruction as a series of
 * chunks — so that single row carries the instruction's FIRST chunk while the
 * model quotes, as often as not, from its fifth. Measured 11/09/2026: testing
 * against the deduped hit recovered 1 row of the 3 that a test against the full
 * document text recovers. The component ran, on the wrong input, and reported a
 * quiet under-effect — the same shape as the reranker that silently did not run
 * for 226 of 300 questions.
 */
export function quotedHits(text: string, cited: Hit[], passages: Hit[]): Hit[] {
  const spans = quotedSpans(text);
  if (spans.length === 0) return [];

  /** Every passage of a document, joined — the text the model was actually shown. */
  const byDoc = new Map<string, string>();
  const key = (h: Hit) => `${h.collection}::${h.code}`;
  for (const p of passages) {
    byDoc.set(key(p), (byDoc.get(key(p)) ?? "") + "\n" + (p.text ?? ""));
  }

  return cited.filter(h => {
    const body = normaliseQuote(byDoc.get(key(h)) ?? h.text ?? "");
    return body.length > 0 && spans.some(s => body.includes(s));
  });
}

/**
 * The closed question behind REFUSAL_QUOTE_CHECK (config.ts has the measurement).
 *
 * The labels are "עונה" / "מסביר" and NOT "כן" / "לא", and that is the whole
 * lesson of the derivation: asked for a one-word yes/no about a question that is
 * itself a yes/no question, the model answers the question. "האם אפשר להאריך את
 * תקופת הפיילוט?" came back "לא" on a text that says exactly that, four times.
 */
const SYSTEM_REFUSAL_QUOTE_CHECK = `אתה מסווג טקסטים. תקבל שאלה וטקסט שנכתב לה על סמך הוראות. אל תענה על השאלה בעצמך.
סווג את הטקסט במילה אחת בלבד:
עונה — אם הטקסט נותן לשאלה עצמה תשובה מתוך ההוראות: ערך, כלל שממנו נובעת התשובה, או תשובה שלילית.
מסביר — אם הטקסט מסביר שההוראות עוסקות במקרה אחר, בגוף אחר או בהקשר אחר, או מפנה למקור אחר, ואינו נותן לשאלה עצמה תשובה.
אל תסביר. אל תוסיף מילה. פלט תקין: עונה`;

/**
 * Does the text of a refusal that goes on to quote an instruction ANSWER the
 * question, or explain why nothing answers it?
 *
 * The quote test (quotedHits) separates "quoted" from "did not quote" and cannot
 * go further: measured 14/09/2026, 7 of 35 correct control refusals quote the
 * very passage they rejected. What separates the two populations is what the
 * text DOES with the quote, which is a reading question, so it is asked of the
 * model - once, only on the few refusals that already passed the quote test.
 *
 * The refusal sentence is removed MECHANICALLY before the call. Telling the
 * model to ignore it did not work: with the sentence left in, two texts that
 * end in a direct answer were classified as not answering.
 *
 * Fails CLOSED. No verdict = no sources, which is today's shipped behaviour
 * with REFUSAL_KEEPS_QUOTED_SOURCES off - the safe side of the 08/09 bug.
 */
export async function checkRefusalQuote(
  transport: LlmTransport,
  question: string,
  text: string,
): Promise<{ verdict: "answers" | "explains" | "failed"; inputTokens: number; outputTokens: number }> {
  // Same lead-stripping as isRefusalText, then the refusal sentence up to its line break.
  let body = text.trim().replace(REFUSAL_LEAD, "");
  if (body.startsWith(REFUSAL_PREFIX)) body = body.replace(/^[^\n]*\n*/, "");
  body = body.trim();
  try {
    const res = await generateWithRetry(transport, SYSTEM_REFUSAL_QUOTE_CHECK, `השאלה: ${question}\n\nהטקסט:\n${body}`, undefined, "quote-check");
    const t = String(res.text ?? "").trim();
    const verdict = /^עונה/.test(t) ? "answers" : /^מסביר/.test(t) ? "explains" : "failed";
    return { verdict, inputTokens: res.inputTokens ?? 0, outputTokens: res.outputTokens ?? 0 };
  } catch {
    return { verdict: "failed", inputTokens: 0, outputTokens: 0 };
  }
}

/**
 * How a citation reads, per collection. TAKAM answers cite a numbered
 * instruction ("הוראה 7.6.1"); an AWS documentation collection has no
 * instruction numbers, so it cites the documentation page instead.
 *
 * The anti-hallucination rules (1, 2, 4, 5) are identical for every corpus on
 * purpose and must not drift apart per collection. What DOES vary is anything
 * describing the corpus itself — what it is, what language it is written in,
 * and how a source is named — because getting those wrong is not a stylistic
 * miss: a prompt that opens "you help find information in TAKAM" while the
 * passages are AWS documentation is telling the model something false about
 * what it is reading, on every single request.
 */
const CITATION_RULE: Record<"instruction" | "page", string> = {
  /**
   * Rule 3 names the document, and it has to name it as the document actually
   * is. The old wording said "(הוראה X.Y.Z)" unconditionally, which was true of
   * the demo corpus and stopped being true the day the full TAKAM arrived: of
   * 957 documents only 411 are instructions, the rest are announcements, forms
   * and annexes carried under their own prefixes (HOD, T, AF, HM).
   *
   * Measured 03/09/2026 on "מה התעריפים לעובדי קבלן". The passage was labelled
   * correctly as "מתוך הודעה HOD.8.2.1.6" — but rule 3 said to write "הוראה", so
   * the model wrote "הוראה 2.1.1", "הוראה 2.1.2", "הוראה 2.1.3", which are the
   * SECTION numbers inside that announcement. The answer itself was right, the
   * rate table was right, and every one of those three labels was wrong.
   *
   * `unverifiedCitations` caught all three and the reader saw a warning — the
   * verifier did its job. But a warning on a correct answer is the expensive
   * kind of noise: it teaches the reader to ignore the warning that matters.
   * The second sentence is therefore the load-bearing one.
   */
  instruction:
    "3. בכל טענה עובדתית ציין את המסמך שממנו היא נלקחה, בדיוק כפי שהוא מסומן בראש הקטע — לדוגמה (הוראה 7.6.1) או (הודעה HOD.8.2.1.6). אל תשנה את סוג המסמך ואל תשנה את המזהה.\n   מספר סעיף בתוך מסמך אינו מזהה של מסמך: כתוב עליו \"סעיף 2.1.1\", לעולם לא \"הוראה 2.1.1\".",
  page: "3. בכל טענה עובדתית ציין את שם עמוד התיעוד שממנו היא נלקחה, בפורמט (עמוד NAME).",
};

/** What the assistant is looking at. The opening line of the prompt. */
const CORPUS_INTRO: Record<"instruction" | "page", string> = {
  instruction:
    'אתה עוזר לאיתור מידע בהוראות תכ"ם (תקנון כספים ומשק) של החשב הכללי במשרד האוצר.',
  page:
    "אתה עוזר לאיתור מידע בתיעוד הרשמי של Amazon Web Services (AWS).",
};

/**
 * The rules that exist only when the sources are in one language and the reader
 * asks in another — which is every AWS question here.
 *
 * Rule 7 is the one that matters most in practice. Left to itself the model
 * produces sentences like "ביצועי startup של עד sub-second", where an English
 * phrase that has a perfectly good Hebrew equivalent is left untranslated
 * simply because it appeared that way in the source. Meanwhile the terms that
 * genuinely MUST stay in English are the identifiers — `SnapStart`,
 * `ProvisionedConcurrency`, a parameter name — because translating those makes
 * them unsearchable and unusable: the reader cannot paste "בו-זמניות מוקצית"
 * into the AWS console.
 *
 * So the rule is not "translate everything" or "keep everything", it is the
 * distinction between the two: identifiers stay, prose becomes Hebrew.
 */
const CROSS_LANGUAGE_RULES =
  "7. הקטעים כתובים באנגלית והתשובה חייבת להיות בעברית תקינה וזורמת. תרגם את התוכן — אל " +
  "תשאיר משפטים או צירופים באנגלית כשיש להם מקבילה עברית טבעית.\n" +
  "8. שמות של שירותים, יכולות, פרמטרים, שדות ופקודות נשארים באנגלית בדיוק כפי שהם " +
  "(Lambda, SnapStart, Provisioned Concurrency, S3 bucket) — הם מזהים, ומי שיתרגם אותם " +
  "לא ימצא אותם בקונסולה. במונח מקצועי שמופיע לראשונה מותר לתת הסבר קצר בסוגריים.\n" +
  "9. אם השאלה עוסקת במושג שהתיעוד מכנה בשם אחר, ענה לפי המשמעות ולא לפי המילה — וציין " +
  "בסוגריים את המונח שבו התיעוד משתמש.";

/**
 * Build the system prompt for one collection.
 *
 * Rule 9 deserves its own note, because it came from a measured failure rather
 * than from taste. The question "מה זה cold start" retrieved the right page at a
 * mediocre score, because AWS does not call it a cold start — the documentation
 * says `execution environment lifecycle`. Semantic retrieval bridged that gap;
 * the prompt now tells the model to bridge it in the other direction too, and to
 * hand the reader the documentation's own word so their next search succeeds.
 */
export function buildSystemPrompt(collection: string = DEFAULT_COLLECTION): string {
  const def = collectionDef(collection);
  const rule3 = CITATION_RULE[def.citation];
  const intro = CORPUS_INTRO[def.citation];
  const crossLanguage = def.language === "he" ? "" : `\n${CROSS_LANGUAGE_RULES}`;
  // ANSWER_SCOPE_FIRST: TAKAM only — see SCOPE_FIRST_RULE.
  const scopeFirst = ANSWER_SCOPE_FIRST && def.citation === "instruction" ? `\n${SCOPE_FIRST_RULE}` : "";
  const closing =
    def.citation === "instruction"
      ? 'זכור: אתה עוזר לאיתור ולניסוח, לא פוסק. האסמכתא המחייבת היא ההוראה המקורית.'
      : "זכור: אתה עוזר לאיתור ולניסוח. האסמכתא המחייבת היא עמוד התיעוד המקורי של AWS.";

  return `${intro}

חוקים מחייבים:
1. ענה אך ורק על סמך הקטעים המצורפים. אל תשתמש בשום ידע חיצוני.
${ANSWER_APPLICABILITY ? RULE2_APPLICABILITY : RULE2_BASE}
${rule3}
4. אל תמציא סכומים, אחוזים, תאריכים או שמות ועדות שאינם כתובים בקטעים.
5. אם הקטעים סותרים זה את זה, ציין זאת במקום להכריע.
6. מסור את כל מה שהקטעים אומרים בנושא השאלה: כל סף, סכום, אחוז, תנאי, חריג וגורם מאשר. צטט מספרים ותאריכים כלשונם.
7. אם הקטע מכיל טבלה (שורות שעמודותיהן מופרדות בטאבים) — שמור על ההתאמה בין כל ערך לכותרת העמודה שלו, ואל תערבב שורות.
8. ענה בעברית ובמבנה ברור. היה מלא ולא מקוצר — עדיף לצטט סעיף שלם מלהשמיט תנאי.${crossLanguage}${scopeFirst}

${closing}${ANSWER_STYLE === "concise" ? CONCISE_STYLE_RULES : ""}`;
}

/** The TAKAM prompt — the default, and what the demo runs on. */
export const SYSTEM_PROMPT = buildSystemPrompt(DEFAULT_COLLECTION);

/** Shown when the system needs the user to pick a topic before answering. */
export const CLARIFY_TEXT = "לא ברור לי למה התכוונת. איזה מהנושאים האלה?";

/**
 * Render the retrieved passages plus the question into one user message.
 *
 * Three details matter here. The passages are NUMBERED, which gives the model a
 * way to refer to them internally and gives us a way to read its answer back
 * against them. Each passage carries its official `keywords` when the catalogue
 * has them — that is the formal vocabulary a citizen would never guess
 * ("ערבויות וביטחונות" for what the user calls "ערבות דיגיטלית"), and showing it
 * to the model lets it answer in the language of the regulation.
 *
 * And every passage is labelled with what it actually IS. A TAKAM passage comes
 * "מתוך הוראה 7.6.1"; an AWS one comes "מתוך עמוד התיעוד lambda-concurrency",
 * and when the search spanned several services it also names the service. This
 * was wrong before — every passage was announced as an "הוראה" regardless — and
 * it is not cosmetic: the label is the model's only cue for how to phrase a
 * citation, and mislabelling the source is a false statement repeated on every
 * request.
 */
export function buildUserMessage(
  question: string,
  hits: Hit[],
  citation: "instruction" | "page" = "instruction",
): string {
  // Naming the service is worth the tokens only when more than one is present:
  // on a single-collection search it is noise repeated on every passage.
  const multiService = new Set(hits.map(h => h.collection)).size > 1;

  const passages = hits
    .map((h, i) => {
      const source =
        citation === "instruction"
          ? `מתוך ${takamKind(h.doctype)} ${h.code}`
          : multiService
            ? `מתוך עמוד התיעוד ${h.code} של ${serviceName(h.collection)}`
            : `מתוך עמוד התיעוד ${h.code}`;
      const header = `[${i + 1}] ${source} — "${h.title}":`;
      const keywords = h.keywords?.length ? `\nמילות מפתח רשמיות: ${h.keywords.join(", ")}` : "";
      return `${header}${keywords}\n${h.text}`;
    })
    .join("\n\n");

  /**
   * The allowed-citation list. Observed in testing: the model correctly refused
   * to invent a FACT, then cited instruction 7.3.4 for a statement it had taken
   * from 7.3.7 — a number that appears nowhere in any retrieved passage. The
   * anti-hallucination rules govern content; nothing was governing the citation
   * label itself, and a wrong instruction number is exactly the error a reader
   * who knows the regulations will catch first.
   *
   * Naming the permitted set explicitly is the cheap half of the fix; the other
   * half is verifying it afterwards in `unverifiedCitations`, because a rule in
   * a prompt is a request and not a guarantee.
   */
  const allowed = [...new Set(hits.map(h => h.code))].join(", ");
  const allowedLabel = citation === "instruction" ? "המספרים הבאים" : "שמות העמודים הבאים";

  return (
    `הקטעים שאותרו:\n\n${passages}\n\n---\n` +
    `כמקור לתשובה מותר לצטט אך ורק את ${allowedLabel}: ${allowed}. אבל אם קטע מפנה בעצמו להוראה או לסעיף אחר, הבא את ההפניה כפי שהיא כתובה וציין שהיא הפניה שמופיעה בהוראה. ` +
    `אם טענה אינה מופיעה באף אחד מהקטעים לעיל — אל תכתוב אותה.\n` +
    `השאלה: ${question}`
  );
}

/** "aws-bedrock-agentcore" reads badly inside a sentence; "AWS Bedrock AgentCore" does not. */
const serviceName = (collection: string): string => collectionDef(collection).label;

/**
 * What to call a TAKAM passage in the prompt.
 *
 * The corpus holds five kinds of document and only one of them is binding
 * policy. A form is a blank to fill in, a message is a point amendment, a tender
 * notice announces a specific procurement and an attached file is an annex —
 * none of them is "הוראה", and announcing them all as one was a false statement
 * repeated on every request once the corpus grew past instructions.
 *
 * The default stays "הוראה" for a passage with no recorded type: the 545 chunks
 * of the original demo collection predate this field, and calling them what they
 * are is right even without the label.
 */
const TAKAM_KINDS: Record<string, string> = {
  Instruction: "הוראה",
  Form: "טופס",
  Message: "הודעה",
  "Tender Notice": "הודעת מכרז",
  "Tender Details": "פרטי מכרז",
  "Tender Maintenance": "עדכון מכרז",
  "Attached File": "נספח",
};
const takamKind = (doctype?: string): string =>
  (doctype && TAKAM_KINDS[doctype]) || "הוראה";

/**
 * Citations in the answer that do not correspond to any retrieved passage.
 *
 * This is the verification half of the citation fix. It exists because of a
 * real failure caught in testing, and it encodes a distinction worth stating in
 * its own right: a model that cannot invent a fact can still invent a label for
 * one. Anything this returns is surfaced to the reader as an explicit warning
 * rather than silently deleted — quietly editing a model's output would hide
 * the very failure the user needs to see.
 */
// Exported 15/09/2026 only so a zero-cost bench can drive it on stored answers.
// `allowPassageRefs` overrides the flag for one call, so ask() can report what the rule cleared.
export function unverifiedCitations(text: string, hits: Hit[], allowPassageRefs = UNVERIFIED_ALLOWS_PASSAGE_REFS): string[] {
  const known = new Set(hits.map(h => h.code));

  /**
   * Only numbers presented AS A CITATION are checked.
   *
   * This used to scan for every instruction-shaped number anywhere in the
   * answer, which flagged things the answer was right to contain: a section
   * number inside the instruction it is quoting ("הוראה 7.6.1, סעיף 2.3"), and
   * a cross-reference the passage itself makes. Rule 6 of the prompt now asks
   * for exactly those, so a verifier that flags them contradicts the prompt and
   * fires on almost every good answer.
   *
   * Rule 3 fixes the citation format as "הוראה X.Y.Z", so requiring that word
   * checks what the rule actually promises. A number introduced as "סעיף" is a
   * pointer inside a document, not a claim about which document was read, and
   * the reader can already see which documents were read from the sources box.
   */
  const found = [...text.matchAll(/הוראה[  ]+(\d+(?:\.\d+){1,3})\b/g)].map(m => m[1]);

  return [...new Set(found)].filter(code => {
    if (known.has(code)) return false;
    // A number that is a prefix or an extension of a retrieved code is almost
    // always an internal section reference ("סעיף 7.3.1.2"), not a claim about
    // a different instruction. Only flag what looks like a genuine mis-citation.
    for (const k of known) {
      if (k.startsWith(`${code}.`) || code.startsWith(`${k}.`)) return false;
    }
    // Bare "7.3" style prefixes are chapter references, not instructions.
    if (code.split(".").length < 3) return false;

    /**
     * A cross-reference the passage itself makes (UNVERIFIED_ALLOWS_PASSAGE_REFS).
     *
     * The comment above this function always said such a reference should not
     * be flagged; until 15/09/2026 nothing here looked at passage TEXT, only at
     * the retrieved codes. g-260 of the gate run is the case: the table in 1.6.3
     * names 3.2.12, the model repeated that, and a correct answer carried an
     * "invented citation" warning. The number must appear with the same digit
     * boundaries citedHits uses, so "3.2.1" is not found inside "3.2.12".
     * A number no passage contains is still flagged - that is the real invention.
     */
    if (allowPassageRefs) {
      const named = new RegExp(`(?<![\\d.])${escapeRe(code)}(?![\\d])(?![.]\\d)`);
      if (hits.some(h => named.test(h.text ?? ""))) return false;
    }

    /**
     * Not everything shaped like an instruction number is one. An Israeli date
     * ("31.12.2026") and a thousands-separated sum ("12.500.000") both match
     * the pattern above, and both were being reported to the reader as INVENTED
     * CITATIONS — on answers that were quoting the regulation correctly.
     *
     * That is worse than cosmetic. This warning is the system's only defence
     * against a model attaching a real fact to the wrong instruction number,
     * and a warning that cries wolf on every date in the corpus is a warning
     * the reader learns to ignore.
     *
     * A TAKAM code is chapter.section.number with each part at most two digits
     * (the corpus tops out at chapter 16). A four-digit part is a year and a
     * three-digit part is a thousands group; neither can be an instruction.
     */
    return code.split(".").every(part => part.length <= 2);
  });
}

// ---------------------------------------------------------------------------
// Error classification (BUGFIX 1)
// ---------------------------------------------------------------------------

/**
 * An HTTP failure from one of the REST transports, carrying the status code so
 * the retry policy can tell "the service is busy" from "your request is wrong".
 * Without the status the caller can only guess, which is exactly how the old
 * loop ended up retrying a malformed request three times.
 */
class LlmHttpError extends Error {
  constructor(readonly status: number, readonly bodyText: string, provider: string) {
    super(`${provider} החזיר שגיאת HTTP ${status}: ${bodyText.slice(0, 400)}`);
    this.name = "LlmHttpError";
  }
}

/**
 * BUGFIX 1 — the old loop caught EVERY error and moved on. Retrying a 400
 * (bad request), a 401 (wrong key) or a validation error is pure waste: the
 * next attempt fails identically, three times slower, and only the last error
 * is ever shown to the user. Retry ONLY the failures that a second attempt can
 * plausibly fix: throttling, server-side faults, and network hiccups.
 */
/**
 * Exported because rerank.ts needs the same judgement. It used to call the
 * transport directly with no retry at all, so a single ThrottlingException
 * ended the rerank silently and the search order answered instead — measured on
 * the 08/09 baseline: 226 of 300 questions were never reranked.
 */
export function isRetryable(err: unknown): boolean {
  const e = err as { name?: string; status?: number; code?: string; $metadata?: { httpStatusCode?: number } };

  // HTTP status, from either a REST transport or the AWS SDK.
  const status = e?.status ?? e?.$metadata?.httpStatusCode;
  if (typeof status === "number") {
    if (status === 429) return true;      // rate limited — capacity, not correctness
    if (status >= 500) return true;       // provider fault
    return false;                          // every other 4xx is OUR bug: fail loudly
  }

  // Bedrock/Smithy exception names that mean "busy" or "temporarily broken".
  const retryableNames = new Set([
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "TimeoutError",
    "AbortError",
  ]);
  if (e?.name && retryableNames.has(e.name)) return true;

  // Node network failures. `fetch` reports these as a TypeError whose cause
  // carries the real code, so both places are checked.
  const networkCodes = new Set(["ECONNRESET", "ETIMEDOUT", "ENOTFOUND", "EAI_AGAIN", "ECONNREFUSED", "UND_ERR_SOCKET"]);
  const cause = (err as { cause?: { code?: string } })?.cause;
  if (e?.code && networkCodes.has(e.code)) return true;
  if (cause?.code && networkCodes.has(cause.code)) return true;
  if (e?.name === "TypeError" && /fetch failed/i.test(String((err as Error)?.message ?? ""))) return true;

  return false;
}

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms));

/** Attempts include the first try. Three attempts, backing off 0.5s then 1s. */
const MAX_ATTEMPTS = 3;

/**
 * Call the transport, retrying only what is worth retrying (see isRetryable).
 * A client error is rethrown on the first attempt, so the person debugging sees
 * the real message instead of the third copy of it.
 */
async function generateWithRetry(
  transport: LlmTransport,
  system: string,
  user: string,
  onToken?: (delta: string) => void,
  /**
   * Which stage this call belongs to, for the retry meter only. It has no
   * effect on the call itself - it is the label a backoff is charged to, so
   * "the answer waited 1.5s" and "the quote check waited 1.5s" stop being the
   * same line in the ledger.
   */
  stage: RetryStage = "answer",
): Promise<GenerateResult> {
  let lastError: unknown;

  /**
   * Stream only when the caller asked for it AND this transport can. Both
   * halves matter: a caller that wants tokens must not crash on a provider
   * that cannot send them, and a caller that does not want tokens must keep
   * the cheaper blocking call. Either way the resolved GenerateResult is the
   * same shape, so nothing downstream branches on which one ran.
   *
   * A retry after a partial stream re-emits the tokens it already sent. That
   * is deliberate: the alternative is suppressing them and leaving the reader
   * looking at half an answer that stopped growing. The UI replaces the text
   * it is showing with the final `text` field when the answer completes, so a
   * duplicated prefix corrects itself rather than persisting.
   */
  const streaming = Boolean(onToken && transport.generateStream);

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    // Start of THIS attempt, so a call that is turned away can still be charged
    // for the time it cost - see RetryLedger.rejectedMs.
    const attemptFrom = Date.now();
    try {
      return streaming
        ? await transport.generateStream!(system, user, onToken!)
        : await transport.generate(system, user);
    } catch (err) {
      lastError = err;
      if (!isRetryable(err)) throw err;                       // BUGFIX 1: fail fast, do not mask
      recordRejectedCall(stage, Date.now() - attemptFrom, (err as Error)?.name ?? "unknown");
      if (attempt === MAX_ATTEMPTS) break;
      const waitMs = 500 * 2 ** (attempt - 1);
      console.error(`   ${transport.id} → ${(err as Error)?.name ?? err} — ניסיון ${attempt}/${MAX_ATTEMPTS}, ממתין ${waitMs}ms`);
      // Measured, not planned: a 500ms timer on a busy event loop is not 500ms,
      // and the point of the ledger is to account for real seconds.
      const waitedFrom = Date.now();
      await sleep(waitMs);
      recordRetryWait(stage, Date.now() - waitedFrom, (err as Error)?.name ?? "unknown");
    }
  }

  throw lastError ?? new Error("שום מודל לא היה זמין");
}

// ---------------------------------------------------------------------------
// Transports (AD-2) — four routes, one interface
// ---------------------------------------------------------------------------

/**
 * Route 1 (DEFAULT): OpenRouter over plain HTTP.
 *
 * No SDK on purpose. The whole protocol is one POST with an OpenAI-shaped body,
 * and Node 24 has `fetch` built in — a dependency here would buy nothing and
 * cost a supply-chain surface. It is also the route that actually works today:
 * every Bedrock generation model in this account is quota-blocked, and this
 * serves the SAME DeepSeek model that `bedrock:deepseek` would serve.
 */
function openRouterTransport(route: string = "openrouter"): LlmTransport {
  const { modelId } = GEN_MODELS[route];

  return {
    id: "openrouter",
    modelId,
    async generate(system: string, user: string): Promise<GenerateResult> {
      if (!OPENROUTER_API_KEY) {
        // Not retryable and not the model's fault — fail with a sentence that
        // says exactly what to do, in the language of the person reading it.
        throw new Error(
          'חסר מפתח OPENROUTER_API_KEY. הוסף אותו ל-poc/.env או ל-.env.local בשורש הריפו, או החלף מסלול עם LLM_ROUTE.',
        );
      }

      const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENROUTER_API_KEY}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: modelId,
          messages: [
            { role: "system", content: system },
            { role: "user", content: user },
          ],
          temperature: 0,             // temperature 0 = as deterministic as possible
          max_tokens: GEN_MAX_TOKENS,
        }),
      });

      if (!res.ok) throw new LlmHttpError(res.status, await res.text(), "OpenRouter");

      const data = (await res.json()) as {
        choices?: { message?: { content?: string }; finish_reason?: string }[];
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      };

      return {
        modelId,
        text: data.choices?.[0]?.message?.content ?? "",
        inputTokens: data.usage?.prompt_tokens ?? 0,
        outputTokens: data.usage?.completion_tokens ?? 0,
        stopReason: data.choices?.[0]?.finish_reason,
      };
    },
  };
}

/**
 * The Bedrock client is created lazily. Constructing it at module load would
 * make every run — including the OpenRouter default, which needs no AWS at all
 * — resolve AWS credentials and fail on a laptop with none configured.
 */
const bedrockClients = new Map<string, BedrockRuntimeClient>();
const getBedrock = (region: string = GEN_REGION): BedrockRuntimeClient => {
  let client = bedrockClients.get(region);
  if (!client) {
    client = new BedrockRuntimeClient({
      region,
      /**
       * Clocks on the socket. Measured 07/09/2026: a momentary DNS failure
       * (getaddrinfo ENOTFOUND bedrock-runtime) held one request open for 35
       * minutes, because the SDK default is to wait forever. connectionTimeout
       * bounds the handshake; requestTimeout is an idle timeout on the socket, so
       * a streaming answer that keeps sending tokens is never cut, only a stall.
       */
      requestHandler: { connectionTimeout: 5_000, requestTimeout: 90_000 },
    });
    bedrockClients.set(region, client);
  }
  return client;
};

/**
 * Rewrite a cross-region inference profile for the geography it is being called
 * from, or refuse to rotate.
 *
 * `us.anthropic.claude-haiku-4-5...` may only be invoked from a US region and
 * `eu.` only from an EU one; crossing them returns a ValidationException whose
 * text ("The provided model identifier is invalid") sounds like a typo rather
 * than a geography mismatch — measured 20/09/2026 against the apac regions,
 * which have no Haiku 4.5 profile at all.
 *
 * Returning undefined means "this model cannot be rotated": a plain ON_DEMAND
 * id such as `deepseek.v3.2` exists in one region only, so moving it is not a
 * throughput trick, it is an error. The caller then pins to GEN_REGION, which
 * is what the code did before this flag existed.
 */
export function regionalModelId(modelId: string, region: string): string | undefined {
  const geo = region.startsWith("us-") ? "us." : region.startsWith("eu-") ? "eu." : undefined;
  if (!geo) return undefined;                                   // apac, and anything new
  const m = /^(us\.|eu\.|apac\.|global\.)(.+)$/.exec(modelId);
  if (!m) return undefined;                                     // not a profile — do not move it
  return `${geo}${m[2]}`;
};

/**
 * Which region the next generation call goes to.
 *
 * A plain incrementing counter, and deliberately not "pick the least loaded" or
 * "remember which region throttled": every call goes through generateWithRetry,
 * so attempt 2 of a throttled call is simply the next call and therefore lands
 * on the next region by itself. That is the behaviour embed.ts arrived at on
 * 30/08/2026 — `EMBED_REGIONS[(i + attempt) % length]` — without a scheduler,
 * and it is worth keeping the simpler shape: state about which ceiling is full
 * would be stale within seconds, since the window refills that fast.
 */
let genRegionCursor = 0;
const nextGenRegion = (): string => {
  if (!GEN_REGIONS.length) return GEN_REGION;
  return GEN_REGIONS[genRegionCursor++ % GEN_REGIONS.length]!;
};

/**
 * The (region, modelId) pair for one call. Falls back to today's behaviour
 * whenever the rotation is off or the model cannot be moved.
 */
export function pickGenTarget(modelId: string): { region: string; modelId: string } {
  if (!GEN_REGIONS.length) return { region: GEN_REGION, modelId };
  const region = nextGenRegion();
  const rewritten = regionalModelId(modelId, region);
  return rewritten ? { region, modelId: rewritten } : { region: GEN_REGION, modelId };
}

/**
 * Routes 2 and 3: Bedrock via the Converse API. Converse is used rather than
 * InvokeModel because it normalises the request/response shape across model
 * families — the same code calls DeepSeek and Claude, which is what makes the
 * route a config value instead of a rewrite.
 */
function bedrockTransport(route: "bedrock:deepseek" | "bedrock:claude"): LlmTransport {
  const { modelId } = GEN_MODELS[route];

  return {
    id: route,
    modelId,
    async generate(system: string, user: string): Promise<GenerateResult> {
      // Chosen per call, not per transport: a retry is a fresh call through
      // this same function, so a throttled attempt automatically asks a
      // different ceiling without any retry code knowing about regions.
      const target = pickGenTarget(modelId);
      recordRegionCall(target.region);
      const res = await getBedrock(target.region).send(
        new ConverseCommand({
          modelId: target.modelId,
          system: [{ text: system }],
          messages: [{ role: "user", content: [{ text: user }] }],
          inferenceConfig: { maxTokens: GEN_MAX_TOKENS, temperature: 0 },
        }),
      );

      return {
        modelId: target.modelId,
        // Every block, not just the first. Converse returns `content` as an
        // ARRAY, and reading [0] alone means any answer whose first block is
        // not plain text — a reasoning block, a guardrail block — comes back
        // empty, while text sitting in later blocks is silently discarded. The
        // Anthropic transport below already joined them; this one did not.
        text: (res.output?.message?.content ?? []).map(c => c.text ?? "").join(""),
        inputTokens: res.usage?.inputTokens ?? 0,
        outputTokens: res.usage?.outputTokens ?? 0,
        stopReason: res.stopReason,
      };
    },

    /**
     * The same call over ConverseStream.
     *
     * ConverseStream, not InvokeModelWithResponseStream, for the same reason
     * `generate` uses Converse: one shape across model families, so the route
     * stays a config value. The IAM policy already allows it —
     * `bedrock:InvokeModelWithResponseStream` is on the query role
     * (deploy.ps1) — so this needed no permission change, only code.
     *
     * Three details that are easy to get wrong:
     *   • The accumulated text MUST be built here and returned, not left to the
     *     caller to reassemble from the deltas it received. The citation check
     *     and the sources box run on the returned text; if the caller kept its
     *     own copy, a dropped delta would leave the warning system checking a
     *     different string from the one on screen.
     *   • usage and stopReason arrive in their OWN events at the end, not with
     *     the text. Reading them from a delta yields zero, which would report
     *     every streamed answer as free and complete — the exact failure the
     *     stopReason field was added to prevent.
     *   • A thrown error mid-stream still reaches generateWithRetry, because
     *     the throw happens inside this promise. Retry behaviour is unchanged.
     */
    async generateStream(system, user, onToken): Promise<GenerateResult> {
      const target = pickGenTarget(modelId);
      recordRegionCall(target.region);
      const res = await getBedrock(target.region).send(
        new ConverseStreamCommand({
          modelId: target.modelId,
          system: [{ text: system }],
          messages: [{ role: "user", content: [{ text: user }] }],
          inferenceConfig: { maxTokens: GEN_MAX_TOKENS, temperature: 0 },
        }),
      );

      let text = "";
      let inputTokens = 0;
      let outputTokens = 0;
      let stopReason: string | undefined;

      for await (const event of res.stream ?? []) {
        const delta = event.contentBlockDelta?.delta?.text;
        if (delta) {
          text += delta;
          onToken(delta);
        }
        if (event.messageStop?.stopReason) stopReason = event.messageStop.stopReason;
        if (event.metadata?.usage) {
          inputTokens = event.metadata.usage.inputTokens ?? 0;
          outputTokens = event.metadata.usage.outputTokens ?? 0;
        }
      }

      return { modelId: target.modelId, text, inputTokens, outputTokens, stopReason };
    },
  };
}

/**
 * Route 4: Anthropic's own API. Kept as the escape hatch for the demo — if both
 * AWS quota and OpenRouter fail on the morning of a demo, one env var
 * moves generation to a third provider without touching a line of logic.
 */
function anthropicTransport(): LlmTransport {
  const { modelId } = GEN_MODELS.anthropic;

  return {
    id: "anthropic",
    modelId,
    async generate(system: string, user: string): Promise<GenerateResult> {
      if (!ANTHROPIC_API_KEY) {
        throw new Error('חסר מפתח ANTHROPIC_API_KEY. הוסף אותו ל-poc/.env, או החלף מסלול עם LLM_ROUTE.');
      }

      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: modelId,
          max_tokens: GEN_MAX_TOKENS,
          temperature: 0,
          system,                                   // Anthropic takes the system prompt as its own field
          messages: [{ role: "user", content: user }],
        }),
      });

      if (!res.ok) throw new LlmHttpError(res.status, await res.text(), "Anthropic");

      const data = (await res.json()) as {
        content?: { text?: string }[];
        usage?: { input_tokens?: number; output_tokens?: number };
        stop_reason?: string;
      };

      return {
        modelId,
        text: data.content?.map(c => c.text ?? "").join("") ?? "",
        inputTokens: data.usage?.input_tokens ?? 0,
        outputTokens: data.usage?.output_tokens ?? 0,
        stopReason: data.stop_reason,
      };
    },
  };
}

/**
 * The switch itself (AD-2). One env var chooses which provider answers; every
 * caller above this line only sees `LlmTransport`. This is the piece that made
 * the Bedrock quota block an inconvenience instead of a dead project.
 */
export function getLlmTransport(route: string = LLM_ROUTE): LlmTransport {
  switch (route) {
    case "openrouter":
    case "openrouter:claude":
      return openRouterTransport(route);
    case "bedrock:deepseek":
    case "bedrock:claude":
      return bedrockTransport(route);
    case "anthropic":
      return anthropicTransport();
    default:
      throw new Error(
        `LLM_ROUTE לא מוכר: "${route}". ערכים אפשריים: ` +
        `openrouter, openrouter:claude, bedrock:deepseek, bedrock:claude, anthropic.`,
      );
  }
}

// ---------------------------------------------------------------------------
// Citation parsing (BUGFIX 2)
// ---------------------------------------------------------------------------

/** Escape a code for safe use inside a RegExp (instruction codes contain dots). */
const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * BUGFIX 2 — which passages did the answer ACTUALLY use?
 *
 * The old code returned every retrieved passage as a source while its own
 * comment claimed otherwise. That is not cosmetic: the sources box is the part
 * a clerk checks against the binding original, and listing a passage the answer
 * never used invites them to "verify" a claim in a document that does not
 * contain it.
 *
 * The check is deliberately simple and explainable: for each retrieved passage,
 * does its code appear as a standalone token in the answer text? Matching the
 * CODE rather than the phrase "הוראה X" is what makes it survive the way models
 * really write ("הוראות 7.6.1 ו-7.6.2"), and the boundary guards stop "7.6"
 * from matching inside "7.6.1".
 */
function citedHits(text: string, hits: Hit[], citation: "instruction" | "page"): Hit[] {
  const seen = new Set<string>();

  /**
   * The best score any passage of a document achieved.
   *
   * Needed because the passages handed to the model are no longer only the
   * ranked hits: withNeighbours adds the paragraphs around them, carrying score
   * 0 to mark them as context. Those are ordered by position, so the first
   * passage of a cited document is often a neighbour — and reporting its 0 in
   * the sources box would tell the reader the answer rests on a passage that
   * matched nothing. The document's real relevance is its best passage.
   */
  const best = new Map<string, number>();
  for (const h of hits) best.set(h.code, Math.max(best.get(h.code) ?? 0, h.score));

  return hits.map(h => ({ ...h, score: best.get(h.code) ?? h.score })).filter(h => {
    if (seen.has(h.code)) return false;                     // one row per document
    const code = escapeRe(h.code);
    const pattern =
      citation === "instruction"
        // The lookahead must still reject a following DIGIT, so that "7.6.1"
        // does not match inside "7.6.1.2" — but it must NOT reject a following
        // period, which is how a citation at the end of a sentence is written:
        // "כאמור בהוראה 7.6.1." was being missed entirely. The fallback below
        // then credited the answer to hits[0], a DIFFERENT instruction, sending
        // a reader to verify a claim against a document that never made it.
        ? new RegExp(`(?<![\\d.])${code}(?![\\d])(?![.]\\d)`)
        : new RegExp(`(?<![\\w-])${code}(?![\\w-])`, "i");    // page slugs, case-insensitive
    if (!pattern.test(text)) return false;
    seen.add(h.code);
    return true;
  });
}

// ---------------------------------------------------------------------------
// The one function the UI and the Lambda call
// ---------------------------------------------------------------------------

export interface AskOptions {
  /**
   * The documents behind the topic the user picked in a clarification.
   *
   * Its presence changes the contract. The person has confirmed what they
   * meant, so retrieval narrows to those documents and the confidence gate no
   * longer applies: that gate exists to stop the SYSTEM from guessing, and once
   * a human has resolved the ambiguity there is nothing left to guess. This is
   * the band in which half of all realistically-worded questions were being
   * refused while the right document sat in the shortlist.
   */
  topicCodes?: string[];
  /** The label the user clicked, recorded so the choice can be learned from. */
  topicLabel?: string;
}

/**
 * The entry point everything calls, and the only place a retry ledger is
 * opened (see retry-meter.ts).
 *
 * The wrapper exists so the accounting happens at exactly one point instead of
 * at the four return sites inside askUnmetered. Whatever the answer turns out
 * to be - refusal, clarification, full answer - it leaves here carrying how
 * much of its own elapsed time was spent asleep waiting for capacity.
 *
 * Behaviour is unchanged: same calls, same backoffs, same text. The fields are
 * evidence, like unverifiedCleared and parentSlackDoc before them, and they are
 * always present so "the meter did not run" cannot be read as "nothing waited".
 */
export async function ask(
  question: string,
  collection: string = DEFAULT_COLLECTION,
  onToken?: (delta: string) => void,
  options: AskOptions = {},
): Promise<Answer> {
  const ledger = newRetryLedger();
  const ans = await runWithRetryLedger(ledger, () => askUnmetered(question, collection, onToken, options));
  ans.retryWaitMs = ledger.waitMs;
  ans.retryCount = ledger.retries;
  ans.retryRejectedMs = ledger.rejectedMs;
  ans.retryRejectedCalls = ledger.rejectedCalls;
  ans.retryByStage = ledger.byStage;
  ans.genRegions = ledger.regions;
  return ans;
}

async function askUnmetered(
  question: string,
  collection: string = DEFAULT_COLLECTION,
  /**
   * Optional. When given, the answer is streamed and this is called with each
   * fragment as it arrives; when omitted, behaviour is byte-for-byte what it
   * was before streaming existed.
   *
   * It is a parameter on THIS function rather than a second `askStream`
   * function, and that is the whole design. Everything around the model call —
   * the refusal gate that never calls it, the neighbour widening, the citation
   * verification, the cost arithmetic — is logic a second copy would be free to
   * drift away from. AD-1 says one core with thin wrappers, and a duplicated
   * ask() would have broken that rule at the most dangerous point in the file.
   *
   * The refusal path returns BEFORE this is ever used, so a refusal streams
   * nothing, costs nothing, and stays the deterministic `if` it has always been.
   */
  onToken?: (delta: string) => void,
  options: AskOptions = {},
): Promise<Answer> {
  const def = collectionDef(collection);

  // Declared members minus the ones never embedded. See resolveMembers.
  const members = await resolveMembers(def);

  /**
   * What was NOT searched, carried on the answer.
   *
   * resolveMembers drops a member with no index, which is the right behaviour —
   * one un-ingested chapter must not take the whole search down. What is not
   * acceptable is doing it quietly: a refusal produced because a chapter was
   * missing is indistinguishable, to the reader, from a refusal produced
   * because the corpus genuinely does not cover the question.
   */
  const declared = def.members ?? [];
  const coverage = {
    searched: members.length || 1,
    declared: declared.length || 1,
    missing: declared.filter(m => !members.includes(m)),
  };
  if (coverage.missing.length) {
    console.warn(`   אזהרה: ${coverage.missing.length} אוספים ללא אינדקס ולא נסרקו — ${coverage.missing.join(", ")}`);
  }

  /**
   * A virtual collection has no index of its own; its members do. The metadata
   * shown to the user ("מאגר המסמכים נכון ל-") is taken from the OLDEST member, not
   * the newest: with several indexes behind one answer, the honest freshness
   * claim is the weakest one. Claiming the newest would be true of one guide
   * and false of the answer.
   */
  const indexedCandidates = members.length
    ? (await Promise.all(members.map(m => loadIndex(m).then(i => i.indexedAt).catch(() => undefined))))
        .filter((v): v is string => Boolean(v))
        .sort()
    : [(await loadIndex(def.id)).indexedAt];

  // A virtual collection whose members all failed to load is not a degraded
  // answer, it is no answer — say so instead of returning a confident-looking
  // reply built on nothing.
  const indexed = indexedCandidates[0];
  if (!indexed) {
    throw new Error(
      `לא נטען אף אינדקס עבור האוסף "${def.id}" — יש להריץ ingest לחברי האוסף לפני השימוש בו.`,
    );
  }

  /**
   * Prepare before retrieving. When the question and the corpus are in the same
   * language this costs nothing and returns instantly; when they are not, it
   * adds an English phrasing of a Hebrew question so retrieval stops paying the
   * cross-lingual penalty. Either way `retrieve` receives plain strings and
   * stays free of any model call.
   */
  const prepared = await prepareQuery(question, def.id);

  // A wider net when the reranker is on: it needs something to choose from.
  // The answer still sees TOP_K passages — rerank() cuts back to it.
  const fetchK = RERANK ? Math.max(RERANK_CANDIDATES, TOP_K) : TOP_K;
  let hits = members.length
    ? await retrieveAcross(prepared.queries, members, fetchK)
    : await retrieve(prepared.queries, def.id, fetchK);

  const retrieved = hits.map(h => ({ code: h.code, title: h.title, score: h.score }));
  const route = LLM_ROUTE;

  if (options.topicCodes?.length) {
    /**
     * The user told us which topic they meant. Narrow to it and answer.
     *
     * The choice is recorded regardless of how the answer turns out, because
     * its value is independent of that: it is a labelled pair — real wording,
     * human-confirmed document — and those are the scarcest thing this project
     * has. The 76-question benchmark had to be model-generated, and its
     * expected answers were wrong often enough to distort the measurement.
     */
    const chosen = hits.filter(h => options.topicCodes!.includes(h.code));
    if (chosen.length) {
      try {
        recordChoice({
          q: question,
          vector: chosen[0].embedding ?? [],
          codes: options.topicCodes,
          label: options.topicLabel ?? "",
          collection: def.id,
        });
      } catch { /* learning is best-effort and must never fail an answer */ }
      hits = chosen;
    }
  } else if (decide(hits, collection) === "clarify") {
    /**
     * Ambiguous: hand the decision back rather than guess.
     *
     * Measured on 76 questions in real user wording — the top document answers
     * 47%, one of the top three answers 66%, and even above 0.55 the first is
     * right only 55% of the time. Those 19 points are not a retrieval problem
     * that more tuning reaches; they are a question only the asker can settle.
     */
    return {
      text: CLARIFY_TEXT,
      refused: false,                              // a choice offered, not a question declined
      sources: [],                                 // nothing answered, nothing cited
      retrieved,
      coverage,
      unverified: [],
      collection: def.id,
      model: GEN_MODELS[route].modelId,
      route,
      indexedAt: indexed,
      usage: { inputTokens: 0, outputTokens: 0, costUSD: 0 },
      clarify: buildClarification(question, hits),
    };
  }

  /**
   * The refusal short-circuit. When retrieval is not confident, the model is
   * NOT called at all — no request, no tokens, no cost. Two reasons, and both
   * are worth saying out loud:
   *   • Determinism. A refusal produced by an `if` is identical every single
   *     run; a refusal produced by asking a model nicely is a probability. KR2
   *     says the system must refuse when the corpus does not cover the
   *     question, and a probability cannot satisfy a "must".
   *   • Money. The out-of-scope question is the cheapest question in the system
   *     instead of the same price as any other.
   * The prompt still carries rule 2 as a second line of defence, for the case
   * where retrieval IS confident but the passages still miss the point.
   */
  if (!options.topicCodes?.length && !isConfident(hits, collection)) {
    return {
      text: REFUSAL_TEXT,
      refused: true,
      refusedAtGate: true,                           // no model, so no reranker either — by design
      sources: [],                                   // a refusal cites nothing, by definition
      retrieved,
      coverage,
      unverified: [],
      collection: def.id,
      model: GEN_MODELS[route].modelId,
      route,
      indexedAt: indexed,
      usage: { inputTokens: 0, outputTokens: 0, costUSD: 0 },
    };
  }

  /**
   * Widen the confident hits into their surrounding paragraphs before writing
   * the prompt. Deliberately AFTER the gate above: adjacency is context, and a
   * passage that arrived because it sits next to a match must not get a vote on
   * whether the corpus covers the question. See withNeighbours in retrieve.ts.
   */
  /**
   * Stage 1½ — see rerank.ts. It runs only once the gate has decided to
   * answer, so a refusal never pays for it, and before the passages are
   * expanded to whole instructions, so the reranker judges the chunks the
   * search actually found rather than 24K characters of context.
   */
  /**
   * Stage 1¼ — parent vote (retrieve.ts). Deliberately here: AFTER the refusal
   * gate, so turning it on cannot change who gets refused, and BEFORE the
   * reranker, so the reranker sees one entry per document instead of three
   * fragments of the same one. See PARENT_VOTE in config.ts.
   */
  const pv = await parentVote(hits);
  hits = pv.hits;

  const rr = await rerank(question, hits);
  hits = rr.hits;

  /**
   * Snapshot of the order the model is about to read in — see Answer.ranked.
   * Taken here, after reranking and before withParentDocuments, because that is
   * the only point where the list is both final and still one entry per passage.
   */
  const ranked = hits.map(h => ({ code: h.code, title: h.title, score: h.score }));

  // Whole instruction when PARENT_CONTEXT is on, otherwise the older window.
  // `parentEvidence` records whether PARENT_TOP_DOC_SLACK ran and what it let in.
  const parentEvidence = { ran: false, doc: null as string | null };
  const passages = PARENT_CONTEXT ? await withParentDocuments(hits, parentEvidence) : await withNeighbours(hits);

  const transport = getLlmTransport(route);
  const systemPrompt = buildSystemPrompt(def.id);
  /**
   * ANSWER_SCOPE_FIRST evidence, read off the prompt actually SENT rather than
   * off the flag: a prompt builder that silently dropped the rule would otherwise
   * read as "the rule did not help". Absent when the flag is off.
   */
  const scopeFirstRule = ANSWER_SCOPE_FIRST ? systemPrompt.includes(SCOPE_FIRST_RULE) : undefined;
  const { modelId, text, inputTokens, outputTokens, stopReason } = await generateWithRetry(
    transport,
    systemPrompt,
    buildUserMessage(question, passages, def.citation),
    onToken,
  );
  // Bedrock reports "max_tokens", OpenRouter "length", the Anthropic API
  // "max_tokens". Any of them means the model was cut off mid-answer, and the
  // reader must be told so rather than shown what looks like a complete answer.
  const truncated = stopReason === "max_tokens" || stopReason === "length";

  // Cost from config, per 1M tokens — never a number typed into this file.
  // The reranker call is the same model at the same price; it is part of what
  // this answer cost, so it is counted here rather than hidden.
  const price = GEN_MODELS[route];
  const totalIn = inputTokens + rr.inputTokens;
  const totalOut = outputTokens + rr.outputTokens;
  const costUSD = (totalIn / 1e6) * price.priceIn + (totalOut / 1e6) * price.priceOut;

  // BUGFIX 2: only the passages the answer actually cites.
  let sources = citedHits(text, passages, def.citation);

  /**
   * Fallback: the model answered, but no code could be parsed out of its text.
   * That happens when it ignores rule 3, or phrases the citation in a way the
   * regex does not know. Returning an empty sources box would be worse than
   * imprecise — it would read as "this answer has no basis at all", when in
   * fact it was grounded in the retrieved passages. So we fall back to the
   * single best-scoring passage: the one the answer is most likely built on,
   * and the smallest honest claim we can make. A refusal is excluded, because
   * a refusal is genuinely sourceless.
   */
  const answered = text.trim().length > 0 && !isRefusalText(text);

  /**
   * A refusal cites nothing. Not "nothing extra" — nothing.
   *
   * The fallback below was only half the problem. A model-written refusal
   * usually explains ITSELF: "I did not find it; the passages I did find
   * concern VAT codes in Sigma (הודעה 2.2.3.1)". citedHits then lifts those
   * numbers out of the text and the UI presents them under "מקורות" — the
   * heading that means "this is what the answer rests on". They are not that.
   * They are what was searched and rejected, and the page already has a place
   * for those: "מה המערכת קראה".
   *
   * Verified 08/09/2026 on "מה שיעור המע\"מ בישראל היום?": three sources on a
   * refusal before this line, none after.
   */
  /**
   * A refusal keeps only the citations it quotes — see quotedHits above for the
   * measurement. OFF by default: it widens what the sources box may show on a
   * refusal, and that is the direction where being wrong is expensive.
   */
  if (!answered) sources = REFUSAL_KEEPS_QUOTED_SOURCES ? quotedHits(text, sources, passages) : [];

  /**
   * And of those, only the ones whose text ANSWERS — see checkRefusalQuote.
   * Runs only when the quote test left something to keep, so the extra call is
   * paid on a few refusals per hundred questions, never on an answer.
   */
  let refusalQuoteCheck: Answer["refusalQuoteCheck"];
  let checkIn = 0, checkOut = 0;
  if (!answered && REFUSAL_QUOTE_CHECK && sources.length > 0) {
    const chk = await checkRefusalQuote(transport, question, text);
    refusalQuoteCheck = chk.verdict;
    checkIn = chk.inputTokens;
    checkOut = chk.outputTokens;
    if (chk.verdict !== "answers") sources = [];
  }

  if (sources.length === 0 && answered && hits.length > 0) {
    sources = [hits[0]];
  }

  /**
   * SOURCES_ADD_QUOTED_DOC: the document a quote was copied from joins the
   * sources, after the fallback so the rule is purely additive (the derivation
   * on stored answers assumed exactly this order). Answers only. The list it
   * added is carried on the answer as evidence that it ran: [] = ran, added
   * nothing; absent = flag off or a refusal.
   */
  let quotedDocsAdded: string[] | undefined;
  if (SOURCES_ADD_QUOTED_DOC && answered) {
    const extra = quotedDocuments(text, sources, passages);
    sources = [...sources, ...extra];
    quotedDocsAdded = extra.map(h => h.code);
  }

  /**
   * The citation warning, and - when UNVERIFIED_ALLOWS_PASSAGE_REFS is on - the
   * numbers the passage-reference rule took OFF it. The second list is the
   * evidence the rule ran (same reason as refusalQuoteCheck): without it a run
   * where the rule never fired and a run where it fired and changed nothing
   * look identical. Costs a second regex pass, no model call.
   */
  const unverified = unverifiedCitations(text, passages);
  const unverifiedCleared = UNVERIFIED_ALLOWS_PASSAGE_REFS
    ? unverifiedCitations(text, passages, false).filter(code => !unverified.includes(code))
    : undefined;

  /**
   * The model refused, but the search had already succeeded — so give the
   * reader the candidates as choices instead of a dead end.
   *
   * Reaching this line at all means the confidence gate passed: an uncovered
   * question was refused earlier, without a model call. So the only questions
   * that get topics here are ones the corpus DOES cover and the model declined
   * to answer — which is the `refused-though-ranked` bucket, six of the
   * twenty-five errors in the 10/09 gate run.
   *
   * `text` is untouched, so the control set still sees the refusal sentence and
   * scores it as a refusal. What changes is only what the reader can do next.
   */
  const refusalTopics = (!answered && REFUSAL_OFFERS_TOPICS && hits.length > 1)
    ? buildClarification(question, hits)
    : undefined;

  return {
    text,
    /**
     * The one place a model-written answer is classified. `answered` above is
     * computed once from the complete text, and the sources, the topics offer
     * and now every consumer downstream all follow from that same boolean —
     * see Answer.refused for the copies this replaces.
     */
    refused: !answered,
    sources: sources.map(h => ({ code: h.code, title: h.title, score: h.score, sourceUrl: h.sourceUrl, doctype: h.doctype })),
    retrieved,
    coverage,
    unverified,
    unverifiedCleared,
    quotedDocsAdded,
    truncated,
    // Absent when the flag is off; null = ran, the top document did not need it.
    parentSlackDoc: parentEvidence.ran ? parentEvidence.doc : undefined,
    collection: def.id,
    model: modelId,
    route,
    indexedAt: indexed,
    reranked: rr.applied,
    rerankReason: rr.reason,
    refusalQuoteCheck,
    scopeFirstRule,
    parentVoted: PARENT_VOTE ? pv.promoted.length : undefined,
    ranked,
    // The quote check is the same model at the same price, so it is counted in.
    usage: {
      inputTokens: totalIn + checkIn,
      outputTokens: totalOut + checkOut,
      costUSD: costUSD + (checkIn / 1e6) * price.priceIn + (checkOut / 1e6) * price.priceOut,
    },
    ...(refusalTopics ? { clarify: refusalTopics } : {}),
  };
}
