/**
 * query-prep.ts — turn ONE user question into the set of queries we search with.
 *
 * Why this file exists, in one measurement. Asking the English AWS collections
 * in Hebrew works, but it works worse, and the numbers say by how much
 * (measured 23/08/2026 over 15 questions against the aws-* collections):
 *
 *   Hebrew question, covered topic     0.398 – 0.640
 *   English question, covered topic    0.582 – 0.710
 *   Hebrew question, NOT covered       0.319 – 0.437
 *
 * The covered and uncovered Hebrew bands very nearly touch — 0.398 against
 * 0.437 — so no threshold separates them cleanly, and two perfectly good
 * questions ("איך עובד חיפוש וקטורי") were being refused. The cause is not the
 * corpus and not the threshold: it is that a Hebrew sentence and an English
 * paragraph land further apart in the embedding space than two English texts
 * do, even in a multilingual model. Cross-lingual similarity is systematically
 * weaker than same-language similarity.
 *
 * The fix is to stop asking across the language gap: translate the question
 * into the corpus's language and search with BOTH strings, scoring every chunk
 * by whichever query matches it better. Two properties make this safe:
 *
 *   • It can only help. The original question is always still one of the
 *     queries, so a bad translation cannot lower a score — max() never drops
 *     below what the untranslated question already scored.
 *   • It cannot block. If the translation call fails, times out, or returns
 *     something unusable, we search with the original alone. A question is
 *     never lost because a helper step broke.
 *
 * This module is deliberately the ONLY place in the question track that is
 * allowed to call a language model before retrieval. `retrieve.ts` states as an
 * invariant that it never talks to a model — that is what makes retrieval
 * inspectable on its own — so the translation happens here, upstream, and
 * `retrieve` still only ever sees plain strings handed to it.
 */

import { collectionDef, QUERY_EXPANSION } from "./config.js";

/** Hebrew block in Unicode. One character is enough to classify the question. */
const HEBREW = /[֐-׿]/;

export const isHebrew = (text: string): boolean => HEBREW.test(text);

/**
 * What the retrieval stage should actually search with.
 *
 * `queries` is what gets embedded. `translation` is surfaced separately so the
 * UI and the CLI can SHOW the user what was searched on their behalf — a silent
 * rewrite of someone's question is the kind of helpfulness that becomes a
 * debugging nightmare the first time it goes wrong.
 */
export interface PreparedQuery {
  original: string;
  queries: string[];
  translation?: string;
  /** Why no translation happened, when none did. Empty on success. */
  note?: string;
}

/**
 * The translation prompt asks for a SEARCH QUERY, not a translation.
 *
 * The distinction is load-bearing. A literal translation of "איך עובד חיפוש
 * וקטורי" is "how does vector search work", which is fine — but a question like
 * "איך נותנים הרשאה לפונקציה לגשת לדלי" translates literally into "how do you
 * give permission to a function to access a bucket", while the documentation
 * says "grant a Lambda function access to an S3 bucket". Asking for the phrasing
 * the docs would use recovers the vocabulary, which is precisely the gap that
 * hurt the scores in the first place.
 */
const TRANSLATE_SYSTEM =
  "You rewrite a user's question into a short search query in the target language, " +
  "using the vocabulary that official technical documentation would use. " +
  "Reply with the query text and nothing else — no quotes, no explanation, no preamble. " +
  "Keep product and service names exactly as they are (Lambda, S3, IAM, OpenSearch, Bedrock).";

/** Guard against a model that ignores "reply with the query only". */
const MAX_QUERY_CHARS = 300;

/**
 * Prepare the queries for one question against one target corpus.
 *
 * `targetLanguage` is taken from the collection definition rather than guessed,
 * so a corpus added later inherits the behaviour without touching this file.
 * When the question is already in the corpus's language, this returns instantly
 * and costs nothing: the common case pays no latency and no tokens.
 */
/**
 * Rewrite a citizen's question in the regulator's own words, and search both.
 *
 * Why this exists, in numbers. Three retrieval improvements in a row — clause
 * chunking, a BM25 leg, stub merging — each correct on its own, moved top-1
 * accuracy on 76 realistically-worded questions by exactly nothing: 45% before,
 * 45% after. The right document was not in the top three for 34% of questions,
 * and neither engine could reach it, because the question and the document
 * simply do not share words. A person writes "פותחים ספק"; the instruction
 * says "רישום ספק במאגר". Semantic search bridges some of that; lexical search
 * none of it; and no amount of tuning either engine helps when the vocabulary
 * itself is the gap.
 *
 * So the question is rewritten — once, cheaply — into the register of the
 * תכ"ם before retrieval, and BOTH phrasings are searched. bestScore() already
 * takes the maximum across phrasings, so a rewrite can only add a way of
 * finding a passage, never remove one.
 *
 * The risk, stated plainly: a rewrite can also add a way of finding a passage
 * for a question the corpus does NOT cover, because the model will happily
 * produce fluent bureaucratic Hebrew about income tax. That is why this was
 * unsafe while a similarity threshold was the refusal gate, and why it is safe
 * now — the gate is the model, which reads the passages and declines. Whether
 * that holds is measured on the 50-question control set after every change to
 * this prompt, not assumed.
 */
const EXPAND_SYSTEM = `אתה מומחה להוראות תכ"ם (תקנון כספים ומשק) של החשב הכללי במשרד האוצר.
תקבל שאלה בשפה יומיומית. נסח אותה מחדש כפי שהיא הייתה מנוסחת בלשון ההוראות עצמן.
כללים מחייבים:
- החלף רק מילים יומיומיות במונח הרשמי המקביל (למשל "כסף" → "תשלום", "ספק חדש" → "הקמת ספק").
- אל תוסיף נושאים, גורמים, מסמכים או הקשרים שלא נזכרו בשאלה. ניחוש על ההוראה הרלוונטית פוגע בחיפוש.
- שמור על אותו נושא ואותו היקף, ובאורך דומה לשאלה המקורית.
החזר בדיוק שני ניסוחים, כל אחד בשורה נפרדת, בלי מספור, בלי הסבר, בלי סימני ציטוט.
אל תענה על השאלה — רק נסח אותה מחדש.`;

const expansionCache = new Map<string, string[]>();

async function expandVocabulary(question: string, collection: string): Promise<PreparedQuery> {
  if (!QUERY_EXPANSION) return { original: question, queries: [question] };

  const cached = expansionCache.get(question);
  if (cached) return { original: question, queries: [question, ...cached] };

  try {
    const { getLlmTransport } = await import("./answer.js");
    const { text } = await getLlmTransport().generate(EXPAND_SYSTEM, question);

    const rewrites = text
      .split(/\r?\n/)
      .map(l => l.replace(/^[\d.\-•*\s]+/, "").replace(/^["'`]|["'`]$/g, "").trim())
      /**
       * Length guard, measured 07/09/2026 on 76 citizen-phrased questions: every
       * rewrite that IMPROVED the ranking was about as long as the question or
       * shorter; every rewrite that BROKE it was 1.5-2x longer, because the extra
       * length was invented context ("זכויות תקציביות", "קבלנים וספקים") that
       * dragged in the wrong instruction. A rewrite that has grown that much is
       * no longer a rephrasing of the question; it is a guess about the answer.
       */
      .filter(l => l.length >= 8 && l.length <= MAX_QUERY_CHARS && l !== question)
      .filter(l => l.length <= Math.max(60, Math.round(question.length * 1.4)))
      .slice(0, 2);

    expansionCache.set(question, rewrites);
    return { original: question, queries: [question, ...rewrites] };
  } catch (err) {
    // An expansion is an optimisation; the question is the product.
    return {
      original: question,
      queries: [question],
      note: `הרחבת השאלה נכשלה (${(err as Error).message.slice(0, 60)}) — החיפוש רץ על הניסוח המקורי בלבד`,
    };
  }
}

export async function prepareQuery(
  question: string,
  collection: string,
): Promise<PreparedQuery> {
  const target = collectionDef(collection).language;
  const asked = isHebrew(question) ? "he" : "en";

  if (asked === target) {
    // Same language — no translation, but the vocabulary gap is still there.
    return expandVocabulary(question, collection);
  }

  try {
    /**
     * Imported at call time, not at module load. `answer.ts` imports the
     * retrieval track, which reaches this module, so a top-level import here
     * would close a cycle. A dynamic import defers the resolution until after
     * both modules have finished evaluating, which is the cheapest correct fix
     * and keeps the transport registry in one place instead of duplicating it.
     */
    const { getLlmTransport } = await import("./answer.js");
    const transport = getLlmTransport();

    const targetName = target === "he" ? "Hebrew" : "English";
    const { text } = await transport.generate(
      TRANSLATE_SYSTEM,
      `Target language: ${targetName}\nQuestion: ${question}`,
    );

    const translation = text.trim().replace(/^["'`]|["'`]$/g, "");

    // An empty or runaway reply is treated as "no translation available"
    // rather than as a query: searching on a paragraph of model chatter would
    // be worse than searching on the original question.
    if (!translation || translation.length > MAX_QUERY_CHARS) {
      return {
        original: question,
        queries: [question],
        note: "התרגום לא היה שמיש — החיפוש רץ על השאלה המקורית בלבד",
      };
    }

    return { original: question, queries: [question, translation], translation };
  } catch (err) {
    // Deliberately swallowed. A translation is an optimisation; a question is
    // the product. The reason is carried back to the caller so it can be shown,
    // never thrown so it can break the answer.
    return {
      original: question,
      queries: [question],
      note: `תרגום השאלה נכשל (${(err as Error).message.slice(0, 80)}) — החיפוש רץ על השאלה המקורית בלבד`,
    };
  }
}
