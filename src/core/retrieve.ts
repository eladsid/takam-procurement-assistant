/**
 * retrieve.ts — the "question track", stage 1: find the relevant passages.
 *
 * This is the part people assume is magic and is actually plain arithmetic.
 * Every chunk was turned into a list of numbers (1024 of them with Cohere
 * Multilingual v3) by an embedding model. The question goes through the SAME
 * model, producing its own 1024 numbers. Then we ask one question of every
 * chunk: does your vector point in the same direction as the question's?
 *
 * That measure is the cosine of the angle between the two vectors:
 * 1.0 = same direction (same meaning), 0 = unrelated. It is the reason the
 * system finds "ערבויות וביטחונות" when the user typed "ערבות דיגיטלית" — it
 * compares meaning, not spelling. Keyword search cannot do that, which is
 * exactly the gap documented in the brief.
 *
 * Two things this module deliberately does NOT do:
 *   - it does not know where the index came from (store.ts owns that), and
 *   - it does not talk to a language model. If the wrong passages come back
 *     from here, no amount of prompt engineering downstream will rescue the
 *     answer, so this stage is worth being able to inspect on its own.
 */

import { loadIndex } from "./store.js";
import { Bm25, rrf } from "./lexical.js";
import { embedAll, assertDims, getEmbedTransport } from "./embed.js";
import { TOP_K, MAX_PER_DOC, NEIGHBOUR_RADIUS, LEXICAL_TABLE_ASSIST, LEXICAL_TABLE_MAX, PARENT_CONTEXT, PARENT_CONTEXT_CHARS, PARENT_TOP_DOC_SLACK, HYBRID_SEARCH, HYBRID_CANDIDATES, PARENT_VOTE, PARENT_VOTE_RATIO, PARENT_VOTE_MIN_CHUNKS, PARENT_VOTE_MIN_TOTAL, PARENT_VOTE_K, minScoreFor } from "./config.js";
import type { Hit, Chunk } from "./types.js";

/**
 * Excerpt markers for documents too long to hand over whole — see
 * withParentDocuments. The marker is plain Hebrew inside the passage text, so it
 * reaches the model exactly where the gap is; no prompt rule is needed for it.
 */
const EXCERPT_GAP = "[…קטעים מההוראה הושמטו כאן — ההוראה ארוכה מכדי להיכלל במלואה…]";
const EXCERPT_RADIUS = Math.max(1, NEIGHBOUR_RADIUS);

/**
 * Cosine similarity between two equal-length vectors.
 *
 * dot(a,b) / (‖a‖·‖b‖) — the dot product measures how much the two vectors
 * agree, and dividing by the lengths removes "this chunk is long" from the
 * comparison so only direction (meaning) is left.
 *
 * Performance note, for honesty rather than for action: a chunk's own norm
 * (‖b‖) never changes once it is embedded, so it could be computed once at
 * ingest time and stored, turning this into a dot product per chunk. We are NOT
 * doing that now. A few thousand vectors of 1024 numbers is a handful of
 * milliseconds per question — far below the network round-trip to the embedding
 * model — and precomputed norms would be one more field that can silently go
 * stale. It becomes worth doing at a corpus size this POC does not have.
 */
export function cosine(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * embedAll is the batching wrapper over whichever transport is configured. It
 * hands back the batch envelope ({ vectors, tokens }); the bare-array form is
 * accepted too, so this module and embed.ts can be written independently
 * without a lockstep merge. One line, and it removes a whole class of
 * "undefined is not iterable" at the seam.
 */
type EmbedAllResult = number[][] | { vectors: number[][]; tokens?: number };
const vectorsOf = (r: EmbedAllResult): number[][] => (Array.isArray(r) ? r : r.vectors);

/**
 * Return the k chunks of ONE collection whose meaning is closest to the
 * question.
 *
 * `collection` is a required argument, not a default: loading the wrong corpus
 * does not throw, it just answers a TAKAM question out of the AWS manual and
 * looks like a bad model. Making the caller name the corpus makes that mistake
 * impossible to commit by accident.
 */
export async function retrieve(
  query: string | string[],
  collection: string,
  k = TOP_K,
  precomputed?: number[][],
): Promise<Hit[]> {
  const index = await loadIndex(collection);

  /**
   * One question may arrive as several phrasings — the original plus a
   * translation into the corpus's language (see query-prep.ts). They are
   * embedded in ONE batch call rather than one call each: the transport
   * already batches, so two phrasings cost one round-trip instead of two.
   */
  const queries = Array.isArray(query) ? query : [query];

  /**
   * "query" matters: Cohere embeds a stored passage and a search query
   * differently, and passing the wrong purpose quietly degrades every score.
   *
   * The fourth argument is the important one. The question is embedded by the
   * model that BUILT this index, read off the envelope — not by whatever
   * EMBED_TRANSPORT currently says. A question and the passages it is compared
   * against have to come from the same vector space, and that space belongs to
   * the collection, not to the process asking.
   *
   * This is what lets TAKAM sit on Titan while the AWS collections stay on the
   * Cohere index they were built with: two models, one query path, no mixing.
   */
  const queryVectors = precomputed ?? vectorsOf(
    await embedAll(queries, "query", undefined, index.embedTransport || undefined),
  );

  /**
   * The hard stop. Vectors from two different embedding models are not
   * comparable — 1024 Cohere numbers against 1536 Gemini numbers is not a
   * "worse" answer, it is meaningless arithmetic — and nothing downstream would
   * notice. If EMBED_TRANSPORT changed since the index was built, we fail here,
   * loudly, in Hebrew, naming the fix.
   */
  try {
    assertDims(queryVectors, index.dims);
  } catch {
    throw new Error(
      `אי-התאמה בין מודל השאלה למודל האינדקס: השאלה הופקה ב-${queryVectors[0]?.length ?? 0} מימדים ` +
      `והאינדקס של "${collection}" נבנה ב-${index.dims} מימדים (${index.embedModel}). ` +
      `וקטורים משני מודלים שונים אינם ברי-השוואה — יש להריץ ingest מחדש לאוסף הזה, ` +
      `או להחזיר את EMBED_TRANSPORT לערך שבו הוא נבנה.`,
    );
  }

  /**
   * The dimension check above is necessary but NOT sufficient, and 30/08/2026 is
   * the day that stopped being theoretical: switching from
   * `cohere.embed-multilingual-v3` to `amazon.titan-embed-text-v2:0` kept the
   * width at exactly 1024. Two different vector spaces, identical shape.
   *
   * So while the corpus was mid-migration — TAKAM rebuilt on Titan, the AWS
   * collections still on Cohere — every dimension check passed and every score
   * was meaningless. Nothing errored. The answers just quietly got worse, which
   * is the single worst failure mode a retrieval system has.
   *
   * The envelope already recorded which model built the index; nothing read it.
   * Comparing the NAME is what actually closes the hole, and it is why the model
   * id is stamped at ingest time rather than inferred at query time.
   */
  const used = getEmbedTransport(index.embedTransport || undefined);
  if (index.embedModel && index.embedModel !== used.modelId) {
    throw new Error(
      `האינדקס של "${collection}" נבנה עם ${index.embedModel}, אבל מנוע ההטמעה ` +
      `"${index.embedTransport}" מצביע כעת על ${used.modelId}. ` +
      `שני המודלים עשויים להחזיר ${index.dims} מימדים, ולכן בדיקת הממדים עוברת — אבל אלה מרחבים ` +
      `וקטוריים שונים, וההשוואה ביניהם מחזירה ציונים חסרי משמעות במקום לזרוק שגיאה. ` +
      `זה קורה כשמזהה המנוע נשאר אותו דבר אבל המודל שמאחוריו הוחלף ב-config. ` +
      `פתרון: הרץ ingest מחדש לאוסף הזה.`,
    );
  }

  // D-3, third attempt — the detection is unchanged and scoped to TABLES only
  // (see tablePhrases); what changed is what it is allowed to affect.
  const phrases = LEXICAL_TABLE_ASSIST ? tablePhrases(queries, index.chunks) : [];

  // Chunks without an embedding are ones a previous ingest never finished. They
  // are skipped rather than scored as 0, so a partial index degrades to "fewer
  // passages" instead of to "wrong passages".
  const ranked: Hit[] = index.chunks
    .filter(c => c.embedding && c.embedding.length === index.dims)
    .map(c => ({ ...c, score: bestScore(queryVectors, c.embedding!) }))
    .sort((a, b) => b.score - a.score);

  /**
   * Fuse in the lexical ranking BEFORE diversifying, so a chunk that only BM25
   * found still competes for a slot and is still subject to the per-document
   * cap. Fusing afterwards would let a lexical hit bypass the cap entirely.
   */
  const fused = HYBRID_SEARCH
    ? await hybridFor(queries, collection, ranked, Math.max(k * 4, 20))
    : ranked;

  const picked = diversify(fused, k);

  /**
   * The table assist, as a SEPARATE channel.
   *
   * A rate table — "בודק תוכנה  99 ₪  לשעה" — carries almost no meaning for an
   * embedding to match, so semantic search cannot reach it and a phrase match
   * is the only handle. That problem is real. What is not acceptable is paying
   * for it with the refusal guarantee, which is what happened when the assist
   * wrote its result into `score`.
   *
   * Here it appends passages instead. They keep their true (low) cosine, they
   * are flagged `lexical`, and isConfident skips them. The model gets to read
   * the table; the decision of whether the corpus covers the question at all
   * stays with the semantic evidence alone.
   */
  if (phrases.length) {
    const already = new Set(picked.map(h => h.id));
    const extras = ranked
      .filter(h => !already.has(h.id) && isTabular(h.text) && phrases.some(p => h.text.includes(p)))
      .slice(0, LEXICAL_TABLE_MAX)
      .map(h => ({ ...h, lexical: true }));
    picked.push(...extras);
  }

  return picked;
}

/**
 * Does this passage look like a table rather than prose?
 *
 * The whole lexical path is restricted to passages that answer yes, and that
 * restriction is the difference between the version that shipped and the one
 * reverted on 03/09/2026.
 *
 * The first attempt promoted ANY passage holding a rare word pair from the
 * question. Measured against the control set, three questions that must be
 * refused stopped being refused — "כמה מס הכנסה משלם שכיר" rose from 0.415 to
 * 0.500 — because prose is full of incidental pairs, and one accidental match
 * was enough to let an unrelated passage through the gate.
 *
 * Scoping to tables works because it matches the actual failure. Semantic
 * search is not weak on tables, it is structurally blind to them: a row reading
 * "בודק תוכנה 99 149" has no meaning to be close to. Prose does not have that
 * problem and therefore does not need the exemption — so it does not get it,
 * and the refusal behaviour that matters most is left exactly as measured.
 */
function isTabular(text: string): boolean {
  const tabs = (text.match(/\t/g) ?? []).length;
  return tabs >= 5;
}

/**
 * Exact word pairs from the question that are rare enough to identify a row.
 *
 * Pairs, not single words: "תוכנה" alone pulls "מה התעריף של בודק תוכנה" to
 * instruction 2.2.20, "רישום עלויות תוכנה" — a measured failure. "בודק תוכנה"
 * is the phrase that appears verbatim in the rate table.
 *
 * Rarity is an ABSOLUTE document count, not a percentage, and that is the
 * second correction from the reverted attempt. A percentage ceiling collapses
 * on a collection with many chunks in few documents: chapter 16 holds 7,415
 * chunks across a few hundred documents, so 2% rejected the real signal
 * ("בודק תוכנה", 5 documents) while admitting one-off noise elsewhere.
 */
function tablePhrases(queries: string[], chunks: Chunk[]): string[] {
  const stop = new Set([
    "מה", "מהו", "מהי", "מהם", "של", "עם", "על", "את", "זה", "יש", "אין", "כמה", "מתי",
    "איך", "למה", "האם", "או", "גם", "כל", "לפי", "בין", "כי", "אם", "כן", "לא", "הם",
    "הן", "הוא", "היא", "אני", "אתה", "צריך", "אפשר", "מותר", "חובה",
  ]);

  const candidates = new Set<string>();
  for (const q of queries) {
    const words = q.replace(/[?!,.;:"'()[\]־–—]/g, " ").split(/\s+/).filter(Boolean);
    for (let i = 0; i + 1 < words.length; i++) {
      const [a, b] = [words[i], words[i + 1]];
      if (stop.has(a) || stop.has(b) || a.length < 3 || b.length < 3) continue;
      candidates.add(`${a} ${b}`);
    }
  }
  if (!candidates.size) return [];

  const MAX_DOCS = 8;
  const kept: string[] = [];
  for (const phrase of candidates) {
    const docs = new Set<string>();
    for (const c of chunks) {
      // Only tabular passages count toward the frequency, for the same reason
      // only they can be promoted: this measures "does this phrase identify a
      // row", not "does this phrase appear in the corpus".
      if (isTabular(c.text) && c.text.includes(phrase)) {
        docs.add(c.code);
        if (docs.size > MAX_DOCS) break;
      }
    }
    if (docs.size > 0 && docs.size <= MAX_DOCS) kept.push(phrase);
  }
  return kept;
}


/**
 * A chunk's score against several phrasings of the same question is the BEST of
 * them, never the average.
 *
 * Averaging would punish exactly the case this exists for: a chunk that the
 * translated query matches perfectly (0.71) and the original Hebrew matches
 * weakly (0.46) averages to 0.585 and can fall under the threshold, even though
 * one of the two queries found it squarely. The phrasings are alternatives, not
 * evidence to be pooled — "did ANY way of asking this find you?" is the
 * question being answered.
 *
 * This is also why adding a phrasing is monotonic and therefore safe: a score
 * can only rise, so a bad translation costs nothing but the tokens.
 */
function bestScore(queryVectors: number[][], embedding: number[]): number {
  let best = -1;
  for (const qv of queryVectors) {
    const s = cosine(qv, embedding);
    if (s > best) best = s;
  }
  return best;
}

/**
 * Search SEVERAL collections at once and merge the results into one ranking.
 *
 * The need is practical: to ask "what is a cold start and how do I reduce it"
 * you should not first have to know that cold starts are a Lambda topic rather
 * than an ECS or a Bedrock one. Picking the corpus is the system's job, and it
 * has a better basis for the choice than the user does — the actual similarity
 * scores.
 *
 * Why merging raw scores across collections is legitimate here: every one of
 * these indexes was built by the SAME embedding model at the same dimension, so
 * a 0.71 in aws-lambda and a 0.71 in aws-s3 mean the same thing. That is not
 * true in general — it would be meaningless against an index built by another
 * model — which is why `retrieve` re-checks dimensions per collection and this
 * function inherits that check rather than skipping it.
 *
 * Each collection is asked for k hits and the merged list is re-diversified, so
 * one exhaustively-documented service cannot sweep every slot: the per-document
 * cap that already protects against one long instruction dominating also
 * protects against one loud corpus dominating.
 */
/**
 * One BM25 index per collection, built on first use and kept.
 *
 * Building it costs a single pass over the chunks — the same text already in
 * memory — and it is reused for every later question. A cold first query pays
 * for it once; every query after that is a map lookup.
 */
const bm25Cache = new Map<string, Bm25>();

async function lexicalFor(collection: string): Promise<Bm25 | null> {
  const hit = bm25Cache.get(collection);
  if (hit) return hit;
  try {
    const idx = await loadIndex(collection);
    const engine = new Bm25(idx.chunks);
    bm25Cache.set(collection, engine);
    return engine;
  } catch {
    return null;
  }
}

/**
 * Hybrid retrieval: the semantic ranking and the lexical ranking, fused.
 *
 * The two engines fail in different places, which is the whole reason to run
 * both. Embeddings compare meaning and go blind exactly where this corpus keeps
 * its answers — "7.6.1" is three numbers, "בודק תוכנה" is a job title in one
 * table, "170" is a rate. Measured on 76 realistically-worded questions, the
 * right instruction never reached the top three for 34% of them.
 *
 * Fusion is by RANK, never by score. Adding a cosine to a BM25 value is
 * comparing two different scales, and one loud lexical match on an incidental
 * word then outranks a genuine semantic one. That is not a hypothetical: it is
 * what happened the last time this project added a lexical leg by combining
 * scores, and it broke the refusal path badly enough to be reverted.
 *
 * What makes it safe THIS time is that the refusal decision has moved. The gate
 * is now the model, which reads the passages; fusion only decides what order it
 * reads them in. A document promoted by a lexical match that turns out to be
 * irrelevant gets read and rejected, instead of silently becoming an answer.
 *
 * The returned `score` stays the SEMANTIC one, deliberately. Downstream code —
 * the confidence floor, the sources box, the clarification bands — is calibrated
 * on cosine, and an RRF number is not on that scale and carries no magnitude at
 * all. A chunk found only by BM25 keeps its true (low) cosine and is ordered by
 * the fusion, which is exactly the separation of "what to read" from "how good
 * is this" that the rest of the file already enforces.
 */
/**
 * A document code named directly in the question — "מה כתוב בהוראה 7.6.1".
 *
 * This is a lookup, not a search, and it deserves to be treated as one. Both
 * engines get it wrong for the same structural reason: an instruction does not
 * print its own number in its body, so neither similarity nor term frequency
 * can connect the question to the document. Measured: 0 of 7.6.1's 108 chunks
 * contain "7.6.1", while 37 chunks of other instructions do.
 *
 * So when the question names something shaped like a code AND that code is a
 * real document in this collection, its chunks are pulled in by identity. A
 * number that matches nothing is ignored — this must not invent a document.
 */
function codesNamedIn(queries: string[], chunks: Hit[]): Set<string> {
  const wanted = new Set<string>();
  for (const q of queries) {
    for (const m of q.matchAll(/\b\d+(?:[.]\d+){1,3}\b/g)) wanted.add(m[0]);
  }
  if (!wanted.size) return new Set();
  const real = new Set<string>();
  for (const c of chunks) if (wanted.has(c.code)) real.add(c.code);
  return real;
}

async function hybridFor(
  queries: string[],
  collection: string,
  semantic: Hit[],
  k: number,
): Promise<Hit[]> {
  const engine = await lexicalFor(collection);
  if (!engine) return semantic;
  const all = (await loadIndex(collection)).chunks as Hit[];
  return fuse(queries, engine, all, semantic, k);
}

/**
 * A BM25 engine over the UNION of several collections, built once and cached.
 *
 * Why a union engine rather than sixteen per-chapter ones: what BM25 treats as
 * a rare word is computed from the documents it was built on. Built per
 * chapter, "מכרז" is common in chapter 7 and rare in chapter 14, so the same
 * word scores differently depending on which chapter a chunk happens to live
 * in, and the sixteen rankings cannot be merged honestly. Built once over the
 * whole corpus, one word has one weight everywhere.
 *
 * The cache holds the PROMISE, not the engine: two questions arriving in the
 * same second share one build instead of each starting their own (the
 * per-collection cache below still has that race; the union is where it
 * would cost twenty seconds, so it is fixed here first).
 */
interface Union { chunks: Hit[]; engine: Bm25; dims: number; embedModel: string; embedTransport: string }
const unionCache = new Map<string, Promise<Union>>();

async function unionOf(members: string[]): Promise<Union> {
  const key = [...members].sort().join("+");
  const cached = unionCache.get(key);
  if (cached) return cached;
  const build = (async (): Promise<Union> => {
    const loaded: { member: string; index: Awaited<ReturnType<typeof loadIndex>> }[] = [];
    for (const member of members) {
      try {
        loaded.push({ member, index: await loadIndex(member) });
      } catch {
        // A member with no index yet contributes nothing — the same tolerance
        // as the per-collection path, so a half-built corpus still answers.
      }
    }
    if (!loaded.length) throw new Error("אף אחד מהאוספים המבוקשים לא נבנה עדיין");
    const first = loaded[0].index;
    for (const { member, index } of loaded.slice(1)) {
      // One vector space or none. This is the same hole the per-collection
      // path closes by model NAME, not by width — see retrieve().
      if (index.dims !== first.dims || index.embedModel !== first.embedModel) {
        throw new Error(
          `האוספים "${loaded[0].member}" ו-"${member}" נבנו במודלים שונים ` +
          `(${first.embedModel} מול ${index.embedModel}) ולכן אינם מרחב וקטורי אחד — ` +
          `יש להריץ ingest מחדש לאחד מהם לפני חיפוש מאוחד.`,
        );
      }
    }
    const chunks = loaded.flatMap(l => l.index.chunks as Hit[]);
    return { chunks, engine: new Bm25(chunks), dims: first.dims, embedModel: first.embedModel, embedTransport: first.embedTransport };
  })();
  unionCache.set(key, build);
  // A failed build must not poison the cache: the next call retries.
  build.catch(() => unionCache.delete(key));
  return build;
}

/** Build the union engine ahead of the first question — see the server warm-up. */
export async function warmUnion(members: string[]): Promise<void> {
  await unionOf(members);
}

/**
 * The fusion itself, independent of WHERE the chunks and the BM25 engine came
 * from: one collection (hybridFor) or the union of many (retrieveUnion).
 */
function fuse(
  queries: string[],
  engine: Bm25,
  all: Hit[],
  semantic: Hit[],
  k: number,
): Hit[] {
  const byId = new Map(semantic.map(h => [h.id, h]));
  const lexical: Hit[] = [];

  /**
   * An explicitly named instruction goes to the FRONT, ahead of both engines.
   * "What does instruction 7.6.1 say" is not a similarity question; the user
   * has already told us the answer's address.
   */
  const named = codesNamedIn(queries, all);
  /**
   * The named instruction as its OWN ranking, in document order.
   *
   * It must NOT be filtered against what the other engines already returned.
   * The semantic list is the whole collection, ranked — nothing is ever
   * "missing" from it, things are merely ranked low. Skipping chunks already
   * present therefore left this list empty on every single query, which is
   * exactly why the lookup appeared to do nothing.
   *
   * Only the opening chunks are taken: they carry the instruction's purpose and
   * definitions, which is what "what does 7.6.1 say" is asking for, and a
   * hundred-chunk document would otherwise swamp the fusion.
   */
  const exact: Hit[] = [];
  for (const code of named) {
    const own = all
      .filter(c => c.code === code)
      .sort((a, b) => a.index - b.index)
      .slice(0, 6);
    // `named` marks a chunk the user asked for BY NAME. The cheap score floor
    // must not apply to it: a low cosine here means only that an instruction
    // does not print its own number in its body, which is true of every
    // instruction and is not evidence of irrelevance.
    for (const c of own) exact.push({ ...(byId.get(c.id) ?? c), named: true } as Hit);
  }
  for (const q of queries) {
    for (const { chunk } of engine.search(q, HYBRID_CANDIDATES)) {
      // Do NOT skip chunks already in `semantic`. That list is the WHOLE
      // collection, ranked — every embedded chunk is in it — so filtering
      // against it discarded 100% of BM25 output on every query and left the
      // lexical leg dead while looking alive. This is the identical mistake
      // already fixed for the named-document list twelve lines above; fixing
      // the symptom there and not the pattern here is why it survived.
      if (lexical.some(l => l.id === chunk.id)) continue;
      // Score it semantically too, so a chunk that arrives via BM25 is still
      // comparable to everything else and cannot masquerade as a strong match.
      // Carry the real cosine when we have one; a BM25-only chunk keeps 0 and
      // is flagged so it cannot masquerade as semantic evidence at the gate.
      const known = byId.get(chunk.id);
      lexical.push(known ?? ({ ...chunk, score: 0, lexical: true } as Hit));
    }
  }
  // Only bail when BOTH extra engines are empty. The first version returned
  // here on `!lexical.length` alone, which threw away the named-instruction
  // list whenever BM25 happened to surface nothing the vectors had missed —
  // and that is the common case for a question like "what does 7.6.1 say",
  // where the whole point is that neither engine can find it by content.
  if (!lexical.length && !exact.length) return semantic;

  /**
   * Three rankings, one per engine, each truncated to its own top-k — and they
   * are ALLOWED to overlap. That is what RRF is for: a chunk both engines rank
   * highly collects two terms and rises, a chunk only one engine likes collects
   * one. Overlap is the signal, not a defect.
   *
   * Two earlier versions got this wrong in opposite directions. The first
   * passed `semantic` and then `[...semantic, ...lexical]`, so every semantic
   * hit sat in two lists and nothing could outrank it. The fix for THAT then
   * over-corrected: it filtered the BM25 list against the semantic list to make
   * them "independent" — but `semantic` is the whole collection ranked, every
   * embedded chunk is in it, so the filter emptied the BM25 list on every
   * query. The lexical leg was dead for a third time, by the same pattern that
   * had already killed the named-document list and the BM25 loop above
   * ("filter against the semantic list"), found 07/09/2026 by reading the code
   * after three retrieval changes in a row left top-1 at exactly 45%.
   *
   * Truncating the semantic list to k makes the two lists symmetric: "absent
   * from the list" means the same thing for both engines, so a dense top hit
   * that BM25 never saw and a BM25 top hit that dense ranked 500th are treated
   * alike — one term each — instead of BM25 hits always carrying two.
   */
  const lists = [semantic.slice(0, k), lexical, ...(exact.length ? [exact] : [])].filter(l => l.length);
  // Stamp the fused position onto each hit — see Hit.fusedRank. Without it the
  // ordering computed here does not survive the next sort.
  /**
   * Re-attach `named` after fusion.
   *
   * rrf keeps the object from the FIRST list a key appeared in, and the
   * semantic list comes first — so the copy that survives is the one WITHOUT
   * the flag, and the exemption in isConfident never fired. The flag is a
   * property of the query ("you asked for this document by number"), not of
   * whichever list happened to carry the object, so it is restored here.
   */
  const namedIds = new Set(exact.map(h => h.id));
  return rrf<Hit>(lists, h => h.id)
    .slice(0, k)
    .map((h, i) => ({ ...h, fusedRank: i, ...(namedIds.has(h.id) ? { named: true } : {}) }));
}

export async function retrieveAcross(
  query: string | string[],
  collections: string[],
  k = TOP_K,
): Promise<Hit[]> {
  /**
   * Embed the question ONCE PER EMBEDDING MODEL, not once per collection.
   *
   * Measured 02/09/2026: asking `takam-all` embedded the same question 16 times
   * — one Bedrock round-trip per member — which put a single user question at
   * ~4 seconds and, worse, spent 16 of the 60 calls-per-minute Titan allows.
   * Four people asking at once would throttle each other with no bug to find.
   *
   * The grouping key is the transport, not "all collections", because the merge
   * is only legitimate between indexes built by the same model: a vector may
   * only be compared with vectors from its own space. Collections that share a
   * transport share one embedding; a collection built by another model gets its
   * own, and `retrieve` still re-checks dimensions and model name per index.
   */
  const byTransport = new Map<string, string[]>();
  for (const c of collections) {
    let key: string;
    try {
      key = (await loadIndex(c)).embedTransport || "";
    } catch {
      // No index built yet. It contributes nothing; retrieve() will say so.
      key = "";
    }
    if (!byTransport.has(key)) byTransport.set(key, []);
    byTransport.get(key)!.push(c);
  }

  const queries = Array.isArray(query) ? query : [query];

  /**
   * ONE search over the whole corpus when every member shares a vector space.
   *
   * The per-collection path below ranks each chapter on its own and then merges
   * the sixteen shortlists by cosine — which quietly throws away the lexical
   * evidence at the merge: a passage BM25 promoted inside chapter 7 arrives with
   * its low cosine and sinks under dense hits from chapter 14. So the hybrid
   * only ever chose WHICH five passages a chapter offered, never the final
   * order. For a corpus that is one body of regulations split into chapters
   * for storage, that split is an artefact, and the honest search is a single
   * dense ranking, a single BM25 over all 28K chunks, and one fusion.
   *
   * The per-collection path stays for genuinely different corpora (the AWS
   * collections were built by another model) and for the single-collection
   * case, where it is the same computation.
   */
  const transports = [...byTransport.keys()];
  if (HYBRID_SEARCH && collections.length > 1 && transports.length === 1 && transports[0] !== "") {
    return retrieveUnion(queries, collections, k);
  }

  const perCollection = (await Promise.all(
    [...byTransport.entries()].map(async ([transport, members]) => {
      let vectors: number[][] | undefined;
      try {
        vectors = vectorsOf(await embedAll(queries, "query", undefined, transport || undefined));
      } catch {
        // Fall through with no precomputed vectors: each member then embeds for
        // itself, which is slower but still answers.
        vectors = undefined;
      }
      return Promise.all(members.map(async c => {
        try {
          return await retrieve(query, c, k, vectors);
        } catch {
          // A member with no index yet must not take the whole search down with
          // it. It contributes nothing and the other members answer.
          return [] as Hit[];
        }
      }));
    }),
  )).flat();

  /**
   * Lexically-assisted passages are held out of the merge and re-appended.
   *
   * They carry their true, low cosine, so a second pass of `diversify` — which
   * ranks purely by score — throws every one of them away. That would leave the
   * assist switched on, firing correctly, and delivering nothing: the feature
   * would be dead while still looking alive in the code.
   */
  const all = perCollection.flat();
  /**
   * Across collections, order by SCORE — never by fusedRank.
   *
   * fusedRank restarts at 0 in every collection, so comparing one collection's
   * rank to another's is comparing two independent sequences. Sorting by it
   * here let a single chapter that happened to produce a fused ranking take
   * every slot ahead of a far better hit elsewhere, which is the exact
   * behaviour diversify() exists to prevent.
   */
  const merged = all.filter(h => !h.lexical);

  /**
   * A document the user named by number goes FIRST across collections, ahead
   * of every score. Its cosine is low by construction — an instruction never
   * prints its own number in its body — so a score-ordered merge pushed 7.6.1
   * (0.233) out of the top five behind unrelated 0.3 hits from other chapters,
   * and the `named` exemption left with it. The user gave us an address; an
   * address is not outranked by resemblance.
   *
   * Everything else is ordered by score, with its per-collection fusedRank
   * REMOVED: those ranks restart at 0 in every collection and are not
   * comparable across them. Leaving them on let one chapter's fused list take
   * every slot, and diversify()'s final sort would have compared them again.
   * The named hits keep a rank so that same sort keeps them in front.
   */
  const named = merged
    .filter(h => h.named)
    .sort((a, b) => (a.fusedRank ?? 0) - (b.fusedRank ?? 0))
    .map((h, i) => ({ ...h, fusedRank: i }));
  const rest = merged
    .filter(h => !h.named)
    .map(h => { const { fusedRank: _drop, ...r } = h; return r as Hit; })
    .sort((a, b) => b.score - a.score);
  const semantic = [...named, ...rest];
  const picked = diversify(semantic, k);

  const already = new Set(picked.map(h => h.id));
  const lexical = all
    .filter(h => h.lexical && !already.has(h.id))
    .sort((a, b) => b.score - a.score)
    .slice(0, LEXICAL_TABLE_MAX);

  return [...picked, ...lexical];
}

/**
 * Take the best k passages, but no more than MAX_PER_DOC from any one document.
 *
 * Pure similarity ranking has a failure mode that only shows up on real data: a
 * long instruction repeats its own subject matter across dozens of passages, so
 * it sweeps every slot and a shorter, equally relevant instruction is pushed
 * out entirely. The observed case: "who approves a tender exemption" filled all
 * five slots from 7.6.1 and never surfaced 7.1.1, the instruction that actually
 * lists the committees.
 *
 * This is a deliberately simple form of result diversification. A second pass
 * walks the same ranking and relaxes the cap, so a question that genuinely has
 * only one relevant source still comes back with k passages rather than two.
 */
/** The whole-corpus search — see the comment at its call site in retrieveAcross. */
async function retrieveUnion(queries: string[], members: string[], k: number): Promise<Hit[]> {
  const union = await unionOf(members);

  const queryVectors = vectorsOf(
    await embedAll(queries, "query", undefined, union.embedTransport || undefined),
  );
  assertDims(queryVectors, union.dims);
  const used = getEmbedTransport(union.embedTransport || undefined);
  if (union.embedModel && union.embedModel !== used.modelId) {
    throw new Error(
      `האיחוד של ${members.length} אוספים נבנה עם ${union.embedModel}, אבל מנוע ההטמעה ` +
      `"${union.embedTransport}" מצביע כעת על ${used.modelId} — מרחבים וקטוריים שונים באותו רוחב. ` +
      `פתרון: הרץ ingest מחדש.`,
    );
  }

  // The same three steps as retrieve(), over the chunks of every member at once.
  const ranked: Hit[] = union.chunks
    .filter(c => c.embedding && c.embedding.length === union.dims)
    .map(c => ({ ...c, score: bestScore(queryVectors, c.embedding!) }))
    .sort((a, b) => b.score - a.score);

  const fused = fuse(queries, union.engine, union.chunks, ranked, Math.max(k * 6, 30));

  // Named documents first, then the FUSED order — not cosine. Re-stamping
  // fusedRank is what makes diversify keep that order.
  const ordered = [...fused.filter(h => h.named), ...fused.filter(h => !h.named)]
    .map((h, i) => ({ ...h, fusedRank: i }));
  const picked = diversify(ordered, k);

  // The table-assist channel, exactly as in retrieve(): shown, never voting.
  const phrases = LEXICAL_TABLE_ASSIST ? tablePhrases(queries, union.chunks) : [];
  if (phrases.length) {
    const already = new Set(picked.map(h => h.id));
    const extras = ranked
      .filter(h => !already.has(h.id) && isTabular(h.text) && phrases.some(p => h.text.includes(p)))
      .slice(0, LEXICAL_TABLE_MAX)
      .map(h => ({ ...h, lexical: true }));
    picked.push(...extras);
  }
  return picked;
}

function diversify(ranked: Hit[], k: number): Hit[] {
  const perDoc = new Map<string, number>();
  const picked: Hit[] = [];

  for (const hit of ranked) {
    if (picked.length >= k) break;
    const key = `${hit.collection}/${hit.code}`;
    const used = perDoc.get(key) ?? 0;
    if (used >= MAX_PER_DOC) continue;
    perDoc.set(key, used + 1);
    picked.push(hit);
  }

  // Backfill in original score order if the cap left us short.
  if (picked.length < k) {
    const chosen = new Set(picked.map(h => h.id));
    for (const hit of ranked) {
      if (picked.length >= k) break;
      if (!chosen.has(hit.id)) picked.push(hit);
    }
  }

  // Fused order wins when there is one; otherwise fall back to score. Sorting
  // unconditionally by score is what silently undid hybrid retrieval.
  return picked.sort((a, b) =>
    (a.fusedRank ?? Infinity) !== (b.fusedRank ?? Infinity)
      ? (a.fusedRank ?? Infinity) - (b.fusedRank ?? Infinity)
      : b.score - a.score);
}

/**
 * Widen each hit into its immediate neighbourhood inside its own document.
 *
 * This is the fix for the most common complaint about this system, and the
 * complaint is precise: the bot missed answers that were inside instructions it
 * had already retrieved. Not the wrong document — the right document, read too
 * narrowly.
 *
 * The arithmetic explains it. A chunk is 900 characters; the search returns 5 of
 * them and at most 2 from any one document. So the model gets about 1,800
 * characters of an instruction that may run 40,000, chosen because those
 * particular sentences resembled the question. But the sentences that RESEMBLE
 * "מה סף הרכישה" are the ones that discuss thresholds in general, while the
 * sentence that ANSWERS it is the row of the table two paragraphs down that says
 * a number and never repeats the topic. Similarity finds the heading; the answer
 * lives just past it.
 *
 * So after ranking, each hit brings its neighbours by position in the source
 * document. They are not re-scored and cannot displace a ranked hit: this
 * changes what the model READS, not what the search FOUND. The confidence gate
 * has already been decided on the ranked hits alone, deliberately — a passage
 * pulled in by adjacency is context, and letting it vote on whether the corpus
 * covers the question would let a weak match drag in filler and call it cover.
 *
 * Passages come back grouped by document and ordered by position within it, so
 * the model reads continuous prose rather than fragments in similarity order.
 * The document whose best hit ranked highest comes first.
 */
/**
 * Widen each hit to the WHOLE INSTRUCTION it came from.
 *
 * Retrieval and reading want different sizes, and this is where they separate.
 * A clause-sized chunk is the right thing to SEARCH — measured, it cut
 * cross-clause contamination from 62% of chunks to 0% — but it is the wrong
 * thing to READ, because a regulation answers a question with its rule, its
 * conditions, its exceptions and its approving authority scattered across
 * neighbouring clauses.
 *
 * So the search unit stays the clause and the context unit becomes the
 * instruction: every chunk of the matched document, in document order. That is
 * what "cut by instruction number" means once you separate the two jobs — and
 * it is strictly more context than the ±1 neighbour window it replaces, which
 * could still stop one clause short of the exception that mattered.
 *
 * The cap is not optional. Instructions run to 30,000 characters and passages
 * are ~91% of what a question costs, so an uncapped expansion would multiply
 * the bill and push the answer toward the far end of a long context. Documents
 * are added whole, best-scoring first, until the budget is spent.
 */
/**
 * How many chunks each document has, per collection. Built once from the index
 * that is already in memory, then reused: the ratio test below needs a
 * denominator, and re-scanning 28,514 chunks for every question to get it would
 * turn a ranking tweak into a measurable latency cost.
 */
const chunkCounts = new Map<string, Map<string, number>>();

async function chunksPerDoc(collection: string): Promise<Map<string, number>> {
  const cached = chunkCounts.get(collection);
  if (cached) return cached;
  const counts = new Map<string, number>();
  try {
    for (const c of (await loadIndex(collection)).chunks) {
      counts.set(c.code, (counts.get(c.code) ?? 0) + 1);
    }
  } catch {
    // No index, no denominator. The ratio test then never fires, and the
    // count test (>= 2 chunks in the list) still works — which is the rule
    // that does the actual work on this corpus.
  }
  chunkCounts.set(collection, counts);
  return counts;
}

/** One group of candidate chunks that all came from the same document. */
export interface ParentVoteInfo {
  code: string;
  collection: string;
  votes: number;
  total: number;
  /** Position (1-based) of this document's best chunk BEFORE the vote. */
  wasAt: number;
}

export interface ParentVoteResult {
  hits: Hit[];
  /** false when the flag is off or nothing qualified — hits are then untouched. */
  applied: boolean;
  promoted: ParentVoteInfo[];
}

/**
 * Parent vote — see PARENT_VOTE in config.ts for why this exists.
 *
 * Three deliberate limits, each one a thing that could go wrong:
 *
 *   - It never touches `score`. The refusal gate reads that number, and a
 *     ranking experiment that can change who gets refused is not a ranking
 *     experiment. Order is expressed by the array's order, exactly as
 *     `fusedRank` does for hybrid search after `diversify` once threw the
 *     fusion away by re-sorting on score.
 *   - It never touches lexical or named hits. A table row was pulled in by a
 *     phrase match and does not vote; a document the user asked for BY NUMBER
 *     stays in front regardless. Same convention as rerank().
 *   - It runs AFTER the confidence gate in answer.ts, so refusals are
 *     bit-identical whether the flag is on or off.
 */
export async function parentVote(hits: Hit[]): Promise<ParentVoteResult> {
  if (!PARENT_VOTE || hits.length < 2) return { hits, applied: false, promoted: [] };

  // Same partition as rerank(), field for field: a hit that is both lexical and
  // named must land in exactly one bucket or the output would carry it twice.
  const extras = hits.filter(h => h.lexical);
  const named = hits.filter(h => !h.lexical && h.named);
  const cand = hits.filter(h => !h.lexical && !h.named);
  if (cand.length < 2) return { hits, applied: false, promoted: [] };

  // Group in list order, so a group's position is its best chunk's position.
  const groups = new Map<string, Hit[]>();
  for (const h of cand) {
    const key = `${h.collection}|${h.code}`;
    const g = groups.get(key);
    if (g) g.push(h); else groups.set(key, [h]);
  }
  if (groups.size === cand.length) return { hits, applied: false, promoted: [] };

  const counts = new Map<string, Map<string, number>>();
  for (const c of new Set(cand.map(h => h.collection))) counts.set(c, await chunksPerDoc(c));

  const promoted: ParentVoteInfo[] = [];
  const ordered: { hit: Hit; joint: number; at: number }[] = [];

  for (const group of groups.values()) {
    const head = group[0];
    const position = cand.indexOf(head) + 1;
    const total = counts.get(head.collection)?.get(head.code) ?? group.length;
    const ratio = total > 0 ? group.length / total : 0;
    /**
     * Both tests, not either — and this is a correction to the plan, forced by
     * a measurement.
     *
     * The plan (following LlamaIndex's AutoMergingRetriever) qualified a
     * document on "2 chunks in the list OR more than half its chunks". On this
     * corpus the second half can essentially never fire — median 12 chunks per
     * document against a 12-candidate list — so the rule collapsed to the count
     * alone, and the count alone is a LENGTH bias wearing agreement's clothes:
     * instruction 7.6.1 has 75 chunks and 6.1.1 has 73, so either one lands two
     * chunks in almost any list by size, not by relevance. Measured on 60 gold
     * questions with retrieval only: the correct instruction moved up 1-5 times
     * and down 7-15 times, at every K from 1 to 60.
     *
     * As AND, the ratio does the job it was meant to do — normalise by document
     * size — so agreement means "most of a short instruction is in view" rather
     * than "a long instruction is long".
     */
    const qualifies =
      group.length >= PARENT_VOTE_MIN_CHUNKS &&
      total >= PARENT_VOTE_MIN_TOTAL &&
      ratio >= PARENT_VOTE_RATIO;

    /**
     * The joint rank: every chunk of the document contributes 1/(K + position),
     * summed. One formula for every document, whether it has siblings in the
     * list or not — a document with one chunk simply has one term, so its order
     * relative to other singletons is unchanged.
     *
     * K is what makes this a nudge instead of a takeover, and it was chosen by
     * measurement, not by taste. The first version promoted every qualifying
     * document to the front of the list as a block, which is the K -> infinity
     * limit of this same formula. Measured on 20 gold questions with retrieval
     * only (no model call, no cost): the CORRECT instruction moved up once and
     * down three times, and lost first place twice while gaining it never. A
     * pair of chunks at positions 7 and 9 was overtaking a single chunk at
     * position 1, and two chunks out of an instruction's 48 is not evidence
     * worth that. With a small K, agreement can lift a document that is already
     * close — which is exactly the measured failure, 34 of 71 misses sitting at
     * rank 2 — and cannot lift one that is far away.
     */
    const joint = group.reduce((sum, h) => sum + 1 / (PARENT_VOTE_K + cand.indexOf(h) + 1), 0);

    if (group.length === 1 || !qualifies) {
      // Nothing to merge: the entry stays exactly as it arrived, scores and all.
      for (const h of group) ordered.push({ hit: h, joint: 1 / (PARENT_VOTE_K + cand.indexOf(h) + 1), at: cand.indexOf(h) });
      continue;
    }

    /**
     * The representative is the group's best-SCORING chunk, not its first in
     * list order: the reranker reads 700 characters of whatever it is handed,
     * and it should read the strongest evidence the group has. Its siblings'
     * ids ride along in `merged` so withParentDocuments still knows they
     * matched.
     */
    const best = group.reduce((a, b) => (b.score > a.score ? b : a));
    promoted.push({ code: head.code, collection: head.collection, votes: group.length, total, wasAt: position });
    ordered.push({
      hit: { ...best, merged: group.map(h => h.id), parentVotes: group.length },
      joint,
      at: position - 1,
    });
  }

  // Nothing qualified: leave the list exactly as it arrived. Collapsing a group
  // that did not earn a promotion would still change how many documents reach
  // the answer stage, and a flag that is "off" must change nothing at all.
  if (!promoted.length) return { hits, applied: false, promoted: [] };

  // Highest joint rank first; ties keep the order the search produced. Named
  // documents stay ahead of all of it and table rows behind, as they were.
  ordered.sort((a, b) => (b.joint - a.joint) || (a.at - b.at));

  return {
    hits: [...named, ...ordered.map(o => o.hit), ...extras],
    applied: true,
    promoted,
  };
}

/**
 * Evidence of what PARENT_TOP_DOC_SLACK did on one call, filled in by
 * withParentDocuments when the caller passes it. `ran` is true whenever the
 * flag is on; `doc` is the code of the top document the slack let in whole, or
 * null when it did not apply (the top document fit, or overflowed by more).
 */
export interface ParentContextEvidence {
  ran: boolean;
  doc: string | null;
}

export async function withParentDocuments(
  hits: Hit[],
  evidence?: ParentContextEvidence,
  slack: number = PARENT_TOP_DOC_SLACK,
): Promise<Hit[]> {
  if (evidence) { evidence.ran = slack > 0; evidence.doc = null; }
  if (!hits.length) return hits;

  const indexes = new Map<string, Hit[]>();
  for (const h of hits) {
    if (indexes.has(h.collection)) continue;
    try {
      indexes.set(h.collection, (await loadIndex(h.collection)).chunks as Hit[]);
    } catch {
      indexes.set(h.collection, []);
    }
  }

  const out: Hit[] = [];
  const seen = new Set<string>();
  let budget = PARENT_CONTEXT_CHARS;

  // Documents in the order their best chunk ranked, so if the budget runs out
  // it is the least relevant instruction that is dropped.
  const docs: { collection: string; code: string; score: number }[] = [];
  for (const h of hits) {
    if (h.lexical) continue;
    if (!docs.some(d => d.collection === h.collection && d.code === h.code)) {
      docs.push({ collection: h.collection, code: h.code, score: h.score });
    }
  }

  for (const [rank, d] of docs.entries()) {
    const all = (indexes.get(d.collection) ?? [])
      .filter(c => c.code === d.code)
      .sort((a, b) => a.index - b.index);
    const size = all.reduce((n, c) => n + c.text.length, 0);

    /**
     * PARENT_TOP_DOC_SLACK (config.ts): the first document only, and only when it
     * overflows the whole budget by a little. An excerpt of the top document can
     * cut out exactly the clause the question asks about (g-021: the fact sat in
     * the gap between the head and the matched window). Budget is clamped at 0,
     * so the documents below still get their matched chunks and nothing else.
     */
    const slackWhole = rank === 0 && slack > 0 && size > budget && size <= PARENT_CONTEXT_CHARS * (1 + slack);
    if (slackWhole && evidence) evidence.doc = d.code;

    if (size <= budget || slackWhole) {
      budget = Math.max(0, budget - size);
      for (const c of all) {
        if (seen.has(c.id)) continue;
        seen.add(c.id);
        // The chunks that actually matched keep their score; the rest are context.
        const hit = hits.find(h => h.id === c.id);
        out.push(hit ?? { ...c, score: 0 });
      }
      continue;
    }

    /**
     * Too large to include whole (79 of the 957 documents; instruction 7.6.1
     * alone is 32K characters against a 24K budget). The first version SKIPPED
     * such a document — "half an instruction reads as a complete one" — and let
     * its matched chunks trail at the end, out of order. Measured 07/09/2026 on
     * "מה כתוב בהוראה 7.6.1?": the model saw two stray clauses and answered,
     * honestly, that it had only seen fragments.
     *
     * Now the document goes in as an EXCERPT: its head (purpose, definitions,
     * the first rules — where a "what does instruction X say" answer lives) plus
     * a window around every matched chunk, in document order. Every gap carries
     * an explicit marker, which answers the original worry: an excerpt that
     * announces its gaps cannot be mistaken for the whole. The excerpt is capped
     * at half the remaining budget so one long instruction cannot starve the
     * documents ranked below it.
     */
    const cap = Math.floor(budget * 0.5);
    /**
     * Every chunk that MATCHED this document — including the siblings that
     * parent vote collapsed into one hit. Without the `merged` ids a promoted
     * three-chunk instruction would be excerpted around one chunk, which is
     * the ranking gain paid for with a worse context.
     */
    const matched = new Set(
      hits
        .filter(h => h.collection === d.collection && h.code === d.code)
        .flatMap(h => h.merged ?? [h.id]),
    );
    const keep = new Set<number>();
    let headChars = 0;
    for (let i = 0; i < all.length; i++) {
      if (headChars + all[i].text.length > cap * 0.6) break;
      headChars += all[i].text.length;
      keep.add(i);
    }
    for (let i = 0; i < all.length; i++) {
      if (!matched.has(all[i].id)) continue;
      const lo = Math.max(0, i - EXCERPT_RADIUS);
      const hi = Math.min(all.length - 1, i + EXCERPT_RADIUS);
      for (let j = lo; j <= hi; j++) keep.add(j);
    }
    let used = 0;
    let prev = -1;
    for (const i of [...keep].sort((a, b) => a - b)) {
      const c = all[i];
      if (seen.has(c.id)) continue;
      // Context chunks respect the cap; the chunks that actually matched always go in.
      if (!matched.has(c.id) && used + c.text.length > cap) continue;
      seen.add(c.id);
      const hit = hits.find(h => h.id === c.id);
      const base = hit ?? { ...c, score: 0 };
      // Always a copy, never a mutation: `all` comes from the shared index cache.
      out.push(i === prev + 1 ? base : { ...base, text: `${EXCERPT_GAP}\n${base.text}` });
      used += c.text.length;
      prev = i;
    }
    if (prev >= 0 && prev < all.length - 1) {
      const last = out[out.length - 1];
      out[out.length - 1] = { ...last, text: `${last.text}\n${EXCERPT_GAP}` };
    }
    budget -= used;
  }

  // Anything not covered above — a matched chunk from a document too large to
  // include whole, and every lexical table row — is appended so nothing that
  // was found is silently dropped.
  for (const h of hits) if (!seen.has(h.id)) { seen.add(h.id); out.push(h); }
  return out;
}

export async function withNeighbours(hits: Hit[], radius = NEIGHBOUR_RADIUS): Promise<Hit[]> {
  if (radius <= 0 || !hits.length) return hits;

  // One index per collection, and loadIndex is cached, so this is a map lookup
  // in the warm case rather than a read.
  const indexes = new Map<string, Map<string, Hit>>();
  for (const h of hits) {
    if (indexes.has(h.collection)) continue;
    try {
      const idx = await loadIndex(h.collection);
      indexes.set(h.collection, new Map(idx.chunks.map(c => [c.id, c as Hit])));
    } catch {
      indexes.set(h.collection, new Map());
    }
  }

  const chosen = new Map<string, Hit>();
  // Rank of the document, by the best-scoring hit it contributed.
  const docRank = new Map<string, number>();

  for (const h of hits) {
    const docKey = `${h.collection}/${h.code}`;
    if (!docRank.has(docKey)) docRank.set(docKey, docRank.size);
    chosen.set(h.id, h);

    const byId = indexes.get(h.collection);
    if (!byId) continue;
    for (let d = -radius; d <= radius; d++) {
      if (d === 0) continue;
      const neighbour = byId.get(`${h.code}#${h.index + d}`);
      // A neighbour already retrieved on its own merit keeps its real score.
      if (!neighbour || chosen.has(neighbour.id)) continue;
      // score 0 marks it as context rather than a match. Nothing downstream
      // ranks on it — the gate already ran — but a reader of the trace can see
      // at a glance which passages the search actually found.
      chosen.set(neighbour.id, { ...neighbour, score: 0 });
    }
  }

  return [...chosen.values()].sort((a, b) => {
    const da = docRank.get(`${a.collection}/${a.code}`) ?? 0;
    const db = docRank.get(`${b.collection}/${b.code}`) ?? 0;
    return da !== db ? da - db : a.index - b.index;
  });
}

/**
 * Does the retrieval clear the bar for being worth answering from?
 *
 * The refusal path is a product feature, not an error case (KR2: every question
 * the corpus does not cover must get an explicit "לא נמצא" instead of an
 * invented answer). A prompt that says "answer only from the sources" is a
 * request; a numeric threshold is a guarantee, and it is the one that survives
 * a model swap. Only the best hit is tested: cosine scores fall off quickly, so
 * an average would let one strong match be dragged under by four weak ones.
 *
 * The threshold is per LANGUAGE, not global: calibrated in Story 1.5 against the
 * deliberately-uncovered Hebrew demo question, and re-measured 23/08/2026 for the
 * English AWS collections, whose score bands sit lower and further apart. Both
 * are measured numbers, not guessed ones — see minScoreFor in config.ts.
 */
export function isConfident(hits: Hit[], collection: string): boolean {
  // Lexically-assisted passages are excluded on purpose: they were added
  // because a phrase matched a table row, which is a reason to SHOW a passage
  // and not evidence that the corpus answers the question. Letting them vote
  // here is precisely the defect that made the system answer 6 of 10
  // out-of-corpus questions.
  // A document the user named by number is relevant by construction: they told
  // us its address. Its cosine is low only because an instruction never prints
  // its own number in its body, which is true of all of them.
  if (hits.some(h => h.named)) return true;

  const semantic = hits.filter(h => !h.lexical);
  // The BEST cosine among the returned passages, not the cosine of whichever
  // passage the fusion put first. Until 07/09/2026 the two were the same thing
  // because the list arrived sorted by score; with the lexical leg alive the
  // list arrives in fused order, and the front slot can legitimately be a
  // passage BM25 promoted on words alone. The gate asks a different question -
  // "does the corpus hold anything semantically close at all?" - and the
  // answer to that is the maximum, wherever it sits.
  return semantic.length > 0 && Math.max(...semantic.map(h => h.score)) >= minScoreFor(collection);
}
