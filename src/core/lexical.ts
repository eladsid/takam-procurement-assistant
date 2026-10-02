/**
 * lexical.ts — the second retrieval engine: BM25 over the same chunks.
 *
 * Why a second engine rather than a better first one. An embedding compares
 * MEANING, and the tokens that matter most in this corpus have almost none:
 * "7.6.1" is three numbers, "בודק תוכנה" is a job title that appears in one
 * table, "170" is a rate. Measured on 76 realistically-worded questions, the
 * right instruction never reached the top three for 34% of them — and those
 * misses cluster exactly on identifier-shaped queries. No amount of tuning a
 * semantic scorer fixes a signal it does not carry.
 *
 * BM25 is the opposite instrument: it knows nothing about meaning and matches
 * rare terms exactly. The two fail in different places, which is the entire
 * reason to run both.
 *
 * The Hebrew handling below is not optional decoration. Hebrew glues its
 * prepositions and conjunctions onto the front of the word — ב, ל, מ, ה, ו, ש,
 * כ — so "במכרז", "למכרז" and "מכרז" are three distinct tokens to a naive
 * index, and a lexical engine that does not strip them is close to useless
 * here. This is a deliberately shallow stemmer, not morphology: it removes
 * prefixes and normalises final letters, and it is honest about being a
 * heuristic rather than pretending to be an analyser.
 */

import type { Chunk } from "./types.js";
import { CONTEXTUAL_RETRIEVAL } from "./config.js";

/** Prefix particles, longest first so "כשה" is stripped before "כ". */
const PREFIXES = ["כשה", "וכש", "מש", "כש", "לכש", "וה", "וב", "ול", "ומ", "וכ", "וש", "ה", "ב", "ל", "מ", "כ", "ש", "ו"];

/** Final forms normalised to their medial equivalents so stems match. */
const FINALS: Record<string, string> = { "ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ" };

/**
 * Words that carry no discriminating power. Kept short on purpose: an
 * over-eager stop list in a legal corpus removes exactly the operators that
 * change a rule's meaning ("אין", "חובה", "רשאי").
 */
const STOP = new Set([
  "של", "את", "על", "עם", "אל", "או", "גם", "כל", "לפי", "בין", "כי", "אם",
  "זה", "זו", "הוא", "היא", "הם", "הן", "אני", "אתה", "יש", "מה", "מהו", "מהי",
  "כמה", "מתי", "איך", "למה", "האם", "אשר", "לא", "כן", "אך", "רק", "וכן",
]);

/**
 * Normalise one token: strip niqqud and gershayim, fold final letters, remove
 * a prefix particle when what remains is still a plausible word.
 *
 * The length guard matters — stripping "ה" from "הוא" leaves "וא", which is
 * noise. Three characters is the shortest Hebrew root worth indexing.
 */
export function normaliseToken(raw: string): string {
  const t = raw
    .replace(/[\u0591-\u05C7]/g, "")   // niqqud and cantillation
    .replace(/["'״׳]/g, "")             // gershayim in acronyms
    .toLowerCase();
  if (!t) return "";
  return t.replace(/[ךםןףץ]/g, c => FINALS[c] ?? c);
}

/**
 * Every form a token should be findable under: itself, and itself with a
 * leading particle removed.
 *
 * BOTH are emitted, and that is the fix for a bug the first version had. It
 * stripped a prefix unconditionally, so "מכרז" became "כרז", "הוראה"
 * became "וראה" and "בודק" became "ודק" — because those words simply
 * BEGIN with a letter that can also be a particle, and nothing in the spelling
 * tells the two apart. Hebrew cannot be stemmed by inspecting one character.
 *
 * Emitting both forms sidesteps the ambiguity rather than guessing at it:
 * "במכרז" indexes as {במכרז, מכרז}, "מכרז" as {מכרז, כרז}; they meet on
 * מכרז and the real word is never destroyed. The cost is a slightly larger
 * index and a little idf dilution — trivial next to splitting a term in two.
 */
function variants(t: string): string[] {
  const out = [t];
  /**
   * Peel repeatedly, keeping every intermediate form.
   *
   * Particles stack: "ומכרז" is ו+מכרז, and stopping after one peel left
   * "מכרז" — whose own leading מ then looked like a particle on the next
   * pass, yielding "כרז". Keeping every step means the true stem is always
   * among the emitted forms even when we cannot tell which step was the right
   * one to stop at. Two peels is enough for Hebrew ("וכשה" is the longest
   * stack that occurs) and bounds the index growth.
   */
  let cur = t;
  for (let depth = 0; depth < 2; depth++) {
    const p = PREFIXES.find(x => cur.length >= x.length + 3 && cur.startsWith(x));
    if (!p) break;
    cur = cur.slice(p.length);
    out.push(cur);
  }
  return [...new Set(out)];
}

/** Split text into normalised terms. Digits and dotted codes are kept whole. */
export function tokenise(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/[^\u0590-\u05FF\w.]+/)) {
    if (!raw) continue;
    // "7.6.1" and "2.4-ג" are identifiers, not sentences — never split or stem.
    if (/^\d[\d.]*$/.test(raw)) { out.push(raw.replace(/\.$/, "")); continue; }
    const t = normaliseToken(raw.replace(/^\.+|\.+$/g, ""));
    if (t.length < 2 || STOP.has(t)) continue;
    for (const v of variants(t)) if (v.length >= 2) out.push(v);
  }
  return out;
}

interface Posting { doc: number; tf: number }

/**
 * A BM25 index built in memory over one collection's chunks.
 *
 * k1 = 1.2 and b = 0.75 are the standard Robertson/Sparck-Jones defaults. They
 * are stated here rather than tuned because tuning them against the current
 * 126-question set would fit the set, not the corpus.
 */
export class Bm25 {
  private postings = new Map<string, Posting[]>();
  private lengths: number[] = [];
  private avgLen = 0;
  private readonly k1 = 1.2;
  private readonly b = 0.75;

  constructor(private chunks: Chunk[]) {
    for (let i = 0; i < chunks.length; i++) {
      // Index the document's IDENTITY, not only its prose.
      //
      // Measured 06/09/2026 and it is stark: none of the 108 chunks of
      // instruction 7.6.1 contains the string "7.6.1", while 37 chunks of OTHER
      // instructions do — because a regulation cites its neighbours and never
      // repeats its own number in its body. Indexing text alone therefore
      // answered "what does instruction 7.6.1 say" with 7.6.3, which cites it.
      // The engine was doing exactly what it was told; it had never been told
      // the number.
      const terms = tokenise(
        // The generated context sentence joins the lexical index too, behind the
        // same flag as the embedding. It carries the topic words a chunk of bare
        // table rows or a bare clause never states — the same gap the identity
        // fields above were added to close, one level up.
        [
          chunks[i].code,
          chunks[i].title,
          (chunks[i].keywords ?? []).join(" "),
          CONTEXTUAL_RETRIEVAL ? (chunks[i].context ?? "") : "",
          chunks[i].text,
        ]
          .filter(Boolean).join(" "),
      );
      this.lengths[i] = terms.length;
      const tf = new Map<string, number>();
      for (const t of terms) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [t, n] of tf) {
        let p = this.postings.get(t);
        if (!p) { p = []; this.postings.set(t, p); }
        p.push({ doc: i, tf: n });
      }
    }
    this.avgLen = this.lengths.reduce((a, x) => a + x, 0) / (this.lengths.length || 1);
  }

  /** Top-k chunk indices by BM25 score, best first. */
  search(query: string, k: number): { chunk: Chunk; score: number }[] {
    const N = this.chunks.length;
    const scores = new Map<number, number>();

    for (const term of new Set(tokenise(query))) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      // Standard BM25 idf, with the +1 that keeps it non-negative for terms
      // appearing in more than half the corpus.
      const idf = Math.log(1 + (N - posting.length + 0.5) / (posting.length + 0.5));
      for (const { doc, tf } of posting) {
        const norm = tf * (this.k1 + 1) /
          (tf + this.k1 * (1 - this.b + this.b * (this.lengths[doc] / this.avgLen)));
        scores.set(doc, (scores.get(doc) ?? 0) + idf * norm);
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([doc, score]) => ({ chunk: this.chunks[doc], score }));
  }
}

/**
 * Reciprocal Rank Fusion — merge ranked lists by POSITION, never by score.
 *
 * `score = Σ 1/(k + rank)` over the lists a document appears in. The point is
 * that it never compares a cosine to a BM25 score: those live on different
 * scales, and adding them is how one loud lexical match on an incidental word
 * ends up outranking a genuine semantic match. That is not hypothetical here —
 * it is what happened when this project last tried a lexical leg by adding
 * scores, and it broke the refusal path badly enough to be reverted.
 *
 * k = 60 is Cormack, Clarke & Buettcher's value from the original paper; its
 * job is to blunt the influence of a single list ranking something first.
 *
 * NOTE what this deliberately does NOT produce: a meaningful absolute score.
 * Fusion discards magnitude, so an RRF number cannot answer "is anything here
 * good enough". That decision belongs to the model, which reads the passages —
 * and keeping the two apart is what makes adding a lexical engine safe now,
 * when it was not before.
 */
export function rrf<T>(lists: T[][], keyOf: (item: T) => string, k = 60): T[] {
  const score = new Map<string, number>();
  const first = new Map<string, T>();

  for (const list of lists) {
    list.forEach((item, i) => {
      const key = keyOf(item);
      score.set(key, (score.get(key) ?? 0) + 1 / (k + i + 1));
      if (!first.has(key)) first.set(key, item);
    });
  }

  return [...score.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => first.get(key)!);
}
