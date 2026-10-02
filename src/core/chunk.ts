/**
 * chunk.ts — turning a whole document into retrievable pieces.
 *
 * Why chunking exists at all: the embedding model turns ONE piece of text into
 * ONE vector. Feed it a 40-page instruction and you get a single vector that
 * means "procurement, generally" — true of every document in the corpus and
 * therefore useless for telling them apart. Feed it a paragraph and you get a
 * vector that means "the exemption committee approves above 50,000 shekels",
 * which is what a question actually looks for. Chunk size is the resolution
 * knob of the whole retrieval system.
 *
 * The strategy here is the one already proven on the Hebrew TAKAM corpus in the
 * first ingest pass, lifted intact: split on blank lines, merge greedily, carry
 * an overlap. Two defects of that first version are fixed, both marked below:
 * the overlap could cut a word in half, and an over-long paragraph escaped
 * un-split. Everything else is deliberately unchanged — it worked.
 *
 * Nothing here calls the network or touches disk, which is what makes it the
 * same code in the Lambda and on the laptop.
 */

import { CHUNK_CHARS, CHUNK_OVERLAP, STRUCTURAL_CHUNKING, MIN_BLOCK_CHARS } from "./config.js";
import type { Chunk } from "./types.js";

// ---------------------------------------------------------------------------
// Sentence splitting — only used to rescue an over-long paragraph
// ---------------------------------------------------------------------------

/**
 * Break a paragraph at sentence ends.
 *
 * The delimiter set is `. ? ! ; :` — each one required to be followed by
 * whitespace. That trailing-whitespace requirement is load-bearing for THIS
 * corpus: instruction numbers ("7.6.1"), sums ("50,000") and version strings
 * are full of periods that are not sentence ends, and none of them has a space
 * after the dot. The lookbehind keeps the punctuation attached to the sentence
 * it closes, so nothing is lost.
 *
 * Colon and semicolon are included because TAKAM enumerates conditions as
 * "יתקיימו התנאים הבאים:" followed by clauses — a natural seam. The Hebrew
 * MAQAF (U+05BE, "־") is deliberately NOT a delimiter: it looks like a dash but
 * it joins two words into one unit ("בית־ספר"), so splitting there would tear a
 * single term apart, which is the opposite of what we want.
 */
function splitSentences(paragraph: string): string[] {
  return paragraph
    .split(/(?<=[.!?;:])\s+/)
    .map(s => s.trim())
    .filter(Boolean);
}

/**
 * Last-resort splitter for a run of text with no sentence punctuation at all —
 * a long table row, a URL list, an un-punctuated Hebrew clause.
 *
 * It cuts on the last space inside the window, so words survive. If the window
 * holds no space (one enormous token) it cuts mid-token: losing a token is
 * strictly better than handing the embedding model text it will silently
 * truncate — silent truncation is invisible at ingest and shows up months later
 * as "why does this rule never come back in search".
 */
function hardSplit(text: string, maxChars: number): string[] {
  const out: string[] = [];
  let rest = text;

  while (rest.length > maxChars) {
    const window = rest.slice(0, maxChars);
    const lastSpace = window.lastIndexOf(" ");
    // Only honour the space if it is reasonably far in; a space at position 3
    // would produce a 3-character chunk and push the whole problem forward.
    const at = lastSpace > maxChars * 0.5 ? lastSpace : maxChars;
    out.push(rest.slice(0, at).trim());
    rest = rest.slice(at).trim();
  }

  if (rest) out.push(rest);
  return out;
}

/**
 * Guarantee that no piece handed to the packer exceeds maxChars.
 *
 * FIX vs. the first ingest pass: it merged paragraphs up to CHUNK_CHARS but
 * never checked a single paragraph against the limit, so one 4,000-character
 * paragraph became one 4,000-character chunk. Cohere Multilingual v3 accepts
 * ~512 tokens and Hebrew runs 2-3 characters per token, so that chunk was
 * truncated by the model — the tail of the paragraph was embedded as if it did
 * not exist, and no error was ever raised.
 */
function toPieces(paragraph: string, maxChars: number): string[] {
  if (paragraph.length <= maxChars) return [paragraph];

  const out: string[] = [];
  let current = "";

  /**
   * Cut on LINE boundaries before sentence boundaries.
   *
   * A block that still contains newlines at this point is a table or a
   * numbered list — extract.ts emits one row or one clause per line. Splitting
   * such a block by sentence would cut through the middle of a row, leaving a
   * value in one chunk and its heading in another. Splitting on lines keeps
   * every row whole, and only a single row too long for the window falls
   * through to the sentence logic below.
   */
  const units = paragraph.includes("\n")
    ? paragraph.split("\n").filter(Boolean)
    : splitSentences(paragraph);

  for (const sentence of units) {
    // A single sentence longer than the window: nothing left but a hard cut.
    if (sentence.length > maxChars) {
      if (current) { out.push(current); current = ""; }
      out.push(...hardSplit(sentence, maxChars));
      continue;
    }
    if (current && current.length + sentence.length + 1 > maxChars) {
      out.push(current);
      current = sentence;
    } else {
      // Rejoin with the separator the unit came from, so a table packed into
      // one chunk still reads as rows rather than as one long line.
      const sep = paragraph.includes("\n") ? "\n" : " ";
      current = current ? `${current}${sep}${sentence}` : sentence;
    }
  }

  if (current) out.push(current);
  return out;
}

/**
 * The tail of the previous chunk, repeated at the head of the next one.
 *
 * FIX vs. the first ingest pass: it used `current.slice(-CHUNK_OVERLAP)`, a
 * blind character cut that routinely started a chunk with "...רות ובטחונות".
 * A half-word is not a word the embedding model knows, so those leading
 * characters are noise inside the vector — small noise, but free to remove.
 * Here the cut snaps FORWARD to the next space, which shortens the overlap by
 * at most one word and never lengthens it.
 *
 * If the tail contains no space at all it is one giant token (a URL, a table
 * row); repeating half of it helps no query, so we return nothing instead.
 */
function overlapTail(text: string, overlap: number): string {
  if (overlap <= 0) return "";
  if (text.length <= overlap) return text;

  const tail = text.slice(-overlap);
  const firstSpace = tail.indexOf(" ");
  return firstSpace === -1 ? "" : tail.slice(firstSpace + 1);
}

// ---------------------------------------------------------------------------
// The chunker
// ---------------------------------------------------------------------------

export interface ChunkOptions {
  /** Hard ceiling on chunk length in characters. Defaults to CHUNK_CHARS. */
  maxChars?: number;
  /** How much of the previous chunk is repeated. Defaults to CHUNK_OVERLAP. */
  overlap?: number;
}

/**
 * Split a document into chunks, preferring paragraph boundaries.
 *
 * Three decisions, in order of how much they matter:
 *
 * 1. Split on BLANK LINES first. A blank line in a Word document is where the
 *    author decided one thought ended, which is a far better boundary than any
 *    character count we could invent. Cutting mid-sentence produces two vectors
 *    that each mean half a rule.
 *
 * 2. Then MERGE greedily up to maxChars. Paragraphs in TAKAM are often two
 *    lines long; one vector per two-line paragraph is too fine — the context a
 *    rule needs ("this applies to exemption committees") sits in the paragraph
 *    before it. Merging keeps related clauses in the same vector.
 *
 * 3. Carry an OVERLAP. A rule that straddles a boundary would otherwise be half
 *    in one chunk and half in the next, and neither half matches the question.
 *    Repeating the tail costs ~20% more storage and embedding tokens and buys
 *    the guarantee that any rule shorter than the overlap appears intact in at
 *    least one chunk. On a 400-chunk corpus that trade is obviously worth it.
 *
 * The output has a hard guarantee the first version did not: every returned
 * string is <= maxChars, so nothing is silently truncated by the model.
 */
/**
 * Split on the document's OWN clause boundaries instead of on a character count.
 *
 * A fixed 900-character window is a guess about where a thought ends. The
 * ministry already made that decision and wrote it into the document: תכ"ם text
 * is numbered "2.", "2.1.", "2.1.3.", and those numbers are boundaries drawn on
 * purpose. Measured across four instructions, 51% of extracted lines carry such
 * a number, with real hierarchical depth (up to five levels) — so the structure
 * is there to be used and we were cutting across it.
 *
 * Why that mattered rather than being merely inelegant: a 900-char window
 * routinely holds the tail of one clause and the head of the next, so its
 * embedding is a blend of two subjects and matches neither cleanly. That is a
 * direct contributor to the measured failure — for 34% of realistically-worded
 * questions the right instruction never reaches the top three at all.
 *
 * The rule: start a new chunk at every TOP-LEVEL clause ("2.", "3."), keep the
 * sub-clauses beneath it together with their parent, and only fall back to the
 * character packer when a single clause is longer than the window. Text before
 * the first number (preamble, definitions) and tables are passed through to the
 * ordinary packer, which already handles them.
 */
const CLAUSE = /^(\d+(?:\.\d+)*)\.?\s/;

/** Depth of a clause number: "2." is 1, "2.1." is 2, "2.1.3." is 3. */
function clauseDepth(line: string): number {
  const m = CLAUSE.exec(line);
  return m ? m[1].split(".").length : 0;
}

/**
 * Group lines into clause blocks. A block is one top-level clause with every
 * sub-clause under it; anything before the first clause is its own block.
 */
function clauseBlocks(text: string): string[] {
  const lines = text.split("\n");
  const blocks: string[] = [];
  let current: string[] = [];

  for (const line of lines) {
    // A depth-1 number starts a new block. Deeper numbers belong to the block
    // above them — splitting "2.1." away from "2." would orphan a condition
    // from the rule it qualifies, which is the exact failure this avoids.
    if (clauseDepth(line) === 1 && current.length) {
      blocks.push(current.join("\n"));
      current = [];
    }
    current.push(line);
  }
  if (current.length) blocks.push(current.join("\n"));
  /**
   * Merge a stub into the block that follows it.
   *
   * Cutting on clause boundaries removed one failure and introduced a worse
   * one. Measured on the rebuilt index: 37% of chunks came out under 100
   * characters and the median fell to 141, because a heading is a depth-1
   * clause exactly like a rule is. "מבוא" and "הגדרות" became standalone
   * chunks 219 and 338 times — six-character vectors that match every question
   * a little and no question well, which is how four out-of-scope questions
   * started clearing the score floor and being answered.
   *
   * A stub is not a passage; it is the label of the passage beneath it.
   * Attaching it forward keeps those words with the text they introduce.
   */
  const merged: string[] = [];
  for (const raw of blocks.map(b => b.trim()).filter(Boolean)) {
    const prev = merged[merged.length - 1];
    if (prev !== undefined && prev.length < MIN_BLOCK_CHARS && prev.length + raw.length <= CHUNK_CHARS) {
      merged[merged.length - 1] = prev + "\n" + raw;
      continue;
    }
    merged.push(raw);
  }
  return merged;
}

export function chunkText(text: string, opts: ChunkOptions = {}): string[] {
  const maxChars = opts.maxChars ?? CHUNK_CHARS;
  const overlap = opts.overlap ?? CHUNK_OVERLAP;

  // Whitespace normalisation inside a paragraph: Word exports are full of soft
  // line breaks and runs of spaces that carry no meaning but do consume the
  // character budget we are trying to spend on actual words.
  const paragraphs = text
    .split(/\n\s*\n/)
    /**
     * Collapse runs of SPACES only. This used to be `/\s+/g`, which also ate
     * every newline and tab — and those are not incidental whitespace here,
     * they are the structure the extractors go to real trouble to produce:
     * extract.ts emits a table row as one tab-separated LINE, for .docx, .xlsx
     * and .pdf alike. Collapsing them turned a price table back into the
     * undelimited word soup that the new .docx extraction exists to prevent,
     * one function after it was built.
     *
     * Newlines inside a paragraph therefore survive to toPieces(), which now
     * treats them as the preferred place to cut.
     */
    .map(p => p.replace(/[ \t]*\n[ \t]*/g, "\n").replace(/ {2,}/g, " ").trim())
    .filter(Boolean);

  /**
   * Prefer the document's own clause boundaries over the character window.
   *
   * Each top-level clause becomes its own chunk, sub-clauses included, so a
   * rule and the conditions that qualify it stay together and a chunk stops
   * being a blend of two unrelated subjects. A clause longer than the window
   * still falls through to the packer below, and a document with no numbering
   * at all (a table annex, a form) is unaffected — clauseBlocks returns it as
   * one block and behaviour is exactly what it was.
   *
   * Disable with STRUCTURAL_CHUNKING=0 to reproduce the previous behaviour.
   */
  const units = STRUCTURAL_CHUNKING
    ? paragraphs.flatMap(p => clauseBlocks(p))
    : paragraphs;

  const out: string[] = [];
  let current = "";

  for (const paragraph of units) {
    // A clause that already fits is emitted whole: packing two short clauses
    // together would undo the boundary we just went to the trouble of finding.
    if (STRUCTURAL_CHUNKING && clauseDepth(paragraph.split("\n")[0]) === 1 && paragraph.length <= maxChars) {
      if (current.trim()) { out.push(current.trim()); current = ""; }
      out.push(paragraph.trim());
      continue;
    }
    for (const piece of toPieces(paragraph, maxChars)) {
      if (current && current.length + piece.length + 1 > maxChars) {
        out.push(current.trim());

        // Shrink the overlap if the incoming piece is nearly a full window on
        // its own. Without this the new chunk would be overlap+piece long and
        // blow the ceiling we just promised to respect.
        const room = maxChars - piece.length - 1;
        const tail = overlapTail(current, Math.min(overlap, Math.max(0, room)));
        current = tail ? `${tail} ${piece}` : piece;
      } else {
        current = current ? `${current} ${piece}` : piece;
      }
    }
  }

  if (current.trim()) out.push(current.trim());
  return out;
}

// ---------------------------------------------------------------------------
// Chunk objects
// ---------------------------------------------------------------------------

/** Everything needed to turn one source document into Chunk records. */
export interface BuildChunksInput {
  /** Corpus id: "takam", "aws-bedrock", ... */
  collection: string;
  /** Citation handle — see deriveCode() in extract.ts. */
  code: string;
  /** Human-readable title shown in the sources box. */
  title: string;
  /** The extracted plain text of the whole document. */
  text: string;
  /** Official catalogue keywords, when the source has them. */
  keywords?: string[];
  /** Ministry document type — see Chunk.doctype. */
  doctype?: string;
  /** Link back to the original document. */
  sourceUrl?: string;
  /** Overrides for this document only; normally left unset. */
  options?: ChunkOptions;
}

/**
 * Build fully-formed Chunk records for one document.
 *
 * Why the metadata is copied onto EVERY chunk instead of being looked up later:
 * after retrieval, a chunk travels alone. The answer step receives five
 * passages ranked by similarity and must be able to say "הוראה 7.6.1" next to
 * each one without a second lookup, and the Lambda has no document table to
 * look anything up in. Denormalising ~100 bytes of metadata per chunk is what
 * makes the vector index a single self-contained JSON file (AD-3).
 *
 * `id` is `${code}#${index}` — stable across re-ingests as long as the source
 * text is unchanged, which is exactly what the resume-from-previous-run logic
 * in the ingest pass keys on to avoid paying for the same embedding twice.
 */
export function buildChunks(input: BuildChunksInput): Chunk[] {
  const parts = chunkText(input.text, input.options ?? {});

  return parts.map((text, index): Chunk => ({
    id: `${input.code}#${index}`,
    collection: input.collection,
    code: input.code,
    title: input.title,
    index,
    text,
    // Omitted rather than set to undefined: these land verbatim in the index
    // JSON, and `"keywords": null` on 76 of 416 documents is noise in a file
    // that is read by a human during the demo.
    ...(input.keywords?.length ? { keywords: input.keywords } : {}),
    ...(input.doctype ? { doctype: input.doctype } : {}),
    ...(input.sourceUrl ? { sourceUrl: input.sourceUrl } : {}),
    // `embedding` is deliberately absent — "not embedded yet" (see types.ts).
  }));
}
