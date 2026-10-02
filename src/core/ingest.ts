/**
 * ingest.ts — the preparation track, as one reusable function.
 *
 * documents (S3 or disk) → text → chunks with citable metadata → embeddings
 * → merged into the collection's vector index.
 *
 * This lives in core/ rather than in a script because it runs from two very
 * different places: a developer types `npm run ingest` to build the whole
 * corpus, and an S3 event fires a Lambda to ingest ONE freshly-uploaded file.
 * Those are the same five steps, so they are the same code — the Lambda and
 * the CLI are thin wrappers that only differ in how they decide which
 * documents to process (AD-1).
 *
 * That single-document path is what makes the demo's closing moment possible:
 * drag instruction 7.10.7 into the bucket during a live demo, and the
 * system can answer questions about it about half a minute later.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { CONTEXTUAL_RETRIEVAL, CORPUS_DIR, LOCAL_DATA_DIR, DEFAULT_COLLECTION, EMBED_MODELS, EMBED_TRANSPORT, EMBED_BATCH } from "./config.js";
import { extractText, deriveCode, titleFor } from "./extract.js";
import { buildChunks } from "./chunk.js";
import { embedAll, estimateTokens } from "./embed.js";
import { listDocumentKeys, readDocument, mergeIntoIndex, readAuxFile } from "./store.js";
import type { Chunk } from "./types.js";

/**
 * Metadata that lives outside the documents themselves: the official title and
 * the catalogue keywords. For TAKAM these come from the manifest written when
 * the corpus was downloaded from the ministry's API — 340 of 416 instructions
 * carry official keywords, and those keywords are the formal vocabulary a user
 * would never guess ("ערבויות וביטחונות" for what people call "ערבות דיגיטלית").
 * Carrying them into each chunk is cheap and measurably helps retrieval.
 */
export interface DocMeta {
  title: string;
  keywords?: string[];
  /** Ministry document type — see Chunk.doctype. */
  doctype?: string;
  sourceUrl?: string;
  version?: string;
}

/**
 * Read the corpus manifest, if there is one. Absence is not an error: an AWS
 * documentation collection has no manifest, and the title is then taken from
 * the document's own first heading. Missing metadata degrades the citation
 * from "instruction 7.6.1 — Exemption from tender" to "instruction 7.6.1",
 * which is worse but still correct.
 */
export async function loadManifest(collection: string): Promise<Map<string, DocMeta>> {
  const meta = new Map<string, DocMeta>();

  const ingestRecords = (raw: string) => {
    const parsed = JSON.parse(raw);
    for (const d of parsed.downloaded ?? parsed.documents ?? []) {
      if (!d?.code) continue;
      /**
       * The public link. The ministry addresses a document by the code as it
       * appears in the catalogue — "H.7.6.1" for an instruction but "T.1.7.2.1"
       * for a form and "HOD.13.5.0.0" for a message. Our own `code` has the
       * "H." stripped (that is what deriveCode does), so prefixing "H." back on
       * unconditionally would build a dead link for every form and message.
       * The manifest's `file` field still carries the original stem, so it is
       * the reliable source; the prefix is only re-added when there is no file.
       */
      const stem = d.file ? String(d.file).replace(/\.(docx|pdf)$/i, "") : `H.${d.code}`;
      meta.set(String(d.code), {
        title: d.title ?? String(d.code),
        keywords: normaliseKeywords(d.keywords),
        doctype: d.doctype ?? undefined,
        sourceUrl: d.url ?? `https://takam.mof.gov.il/document/${stem}`,
        version: d.version != null ? String(d.version) : undefined,
      });
    }
  };

  /**
   * Local copies first — authoritative during development and free to read.
   *
   * BOTH files are read, narrowest LAST so it wins, and this ordering is the
   * fix for a defect measured on 02/09/2026: only `manifest.json` was read and
   * the function returned immediately after it. That file is the ORIGINAL
   * 18-instruction demo manifest. `manifest-all.json` — the whole corpus the
   * downloader writes — was never opened.
   *
   * The consequence was invisible because nothing errored: 393 of 411
   * instructions were indexed with no title, no official keywords and no source
   * link. Only 10.4% of chunks carried a real title; the rest fell back to
   * showing their own code. The keywords matter most of the three — answer.ts
   * hands them to the model as the formal vocabulary a citizen would never
   * guess ("ערבויות וביטחונות" for what they call "ערבות") — and 393
   * instructions were reaching the model without them.
   *
   * A missing file is still not an error: an AWS collection has neither.
   */
  let readAny = false;
  for (const name of ["manifest-all.json", "manifest.json"]) {
    const localPath = join(CORPUS_DIR, name);
    if (!existsSync(localPath)) continue;
    try {
      ingestRecords(readFileSync(localPath, "utf-8"));
      readAny = true;
    } catch (err) {
      console.warn(`   אזהרה: ${name} מקומי לא נקרא (${(err as Error).message}) — ממשיך בלעדיו`);
    }
  }
  if (readAny) return meta;

  /**
   * In the cloud the manifest rides along next to the documents.
   *
   * BOTH names are tried, widest first, for the same reason as the local path
   * above: reading only `manifest.json` is what cost 393 instructions their
   * titles and keywords. That was fixed locally on 02/09/2026 and this branch
   * was left behind, so a cloud ingest still indexed every document with no
   * official title, no catalogue keywords and no source link — the identical
   * defect, surviving in the half nobody was testing.
   */
  for (const name of ["manifest-all.json", "manifest.json"]) {
    const remote = await readAuxFile(collection, name);
    if (!remote) continue;
    try {
      ingestRecords(remote.toString("utf-8"));
      return meta;
    } catch (err) {
      console.warn(`   אזהרה: ${name} ב-S3 לא נקרא (${(err as Error).message}) — ממשיך בלעדיו`);
    }
  }

  return meta;
}

/**
 * The TAKAM API returns keywords as one comma-separated string; other sources
 * may already use an array. Normalise both into the same shape rather than
 * making every downstream reader handle two cases.
 */
function normaliseKeywords(raw: unknown): string[] | undefined {
  if (!raw) return undefined;
  const list = Array.isArray(raw)
    ? raw.map(String)
    : String(raw).split(/[,;\n]/);
  const cleaned = list.map(k => k.trim()).filter(Boolean);
  return cleaned.length ? cleaned : undefined;
}

export interface IngestResult {
  collection: string;
  documents: number;
  chunks: number;
  embedded: number;
  reused: number;
  tokens: number;
  costUSD: number;
  elapsedMs: number;
  failures: { key: string; error: string }[];
}

export interface IngestOptions {
  collection?: string;
  /** Specific object keys to ingest. Omit to ingest everything in the collection. */
  keys?: string[];
  /**
   * Reuse embeddings already in the index for chunks whose text is unchanged.
   * On by default: re-embedding identical text is money spent for no change,
   * and it is what makes a re-run after a crash cheap.
   */
  reuseExisting?: boolean;
  onLog?: (line: string) => void;
}

/**
 * USD per 1M input tokens, read from the model actually in use.
 *
 * It used to be a single module constant, and the constant was wrong: it held
 * Cohere's $0.10 while every embedding was being produced by Titan at $0.02, so
 * every reported cost was 5x the truth. An env override stays available for a
 * price change, but the default now follows the transport instead of having to
 * be remembered whenever the transport changes.
 *
 * Note also that Cohere returns no token count in its response, so the tally
 * feeding this is `estimateTokens`, not an invoice.
 */
const EMBED_PRICE_PER_1M = Number(
  process.env.EMBED_PRICE_PER_1M ?? EMBED_MODELS[EMBED_TRANSPORT]?.pricePer1M ?? 0.1,
);

export async function ingest(options: IngestOptions = {}): Promise<IngestResult> {
  const collection = options.collection ?? DEFAULT_COLLECTION;
  const log = options.onLog ?? ((line: string) => console.log(line));
  const started = Date.now();
  const failures: IngestResult["failures"] = [];

  const keys = options.keys?.length ? options.keys : await listDocumentKeys(collection);
  if (!keys.length) {
    throw new Error(`לא נמצאו מסמכים לאוסף "${collection}" — יש להעלות מסמכים לפני בליעה`);
  }

  const meta = await loadManifest(collection);
  log(`נמצאו ${keys.length} מסמכים לאוסף "${collection}"`);

  // ---- 1..3: read, extract, chunk -----------------------------------------
  // A failure here is per-document on purpose (FR-1.4). One malformed file must
  // not abandon the other 415 — the run reports it and carries on.
  const chunks: Chunk[] = [];
  let documents = 0;

  for (const key of keys) {
    try {
      const { fileName, bytes } = await readDocument(key);
      const { text } = await extractText(fileName, bytes);
      const code = deriveCode(fileName, collection);
      const known = meta.get(code);

      // titleFor wants the RAW source, not the extracted prose: extraction
      // flattens Markdown headings into ordinary lines, so the `# Title` that
      // names an AWS documentation page is gone by then. Only Markdown needs
      // this — a .docx title always comes from the manifest.
      const rawText = /\.(md|markdown|txt)$/i.test(fileName) ? bytes.toString("utf-8") : undefined;
      const title = titleFor({
        fileName,
        collection,
        code,
        rawText,
        catalogTitle: known?.title,
      });

      const built = buildChunks({
        collection,
        code,
        title,
        text,
        keywords: known?.keywords,
        doctype: known?.doctype,
        sourceUrl: known?.sourceUrl,
      });

      chunks.push(...built);
      documents++;
      log(`  ${code.padEnd(9)} ${String(built.length).padStart(4)} קטעים  ${title}`);
    } catch (err) {
      const message = (err as Error)?.message ?? String(err);
      failures.push({ key, error: message });
      log(`  ✗ ${key} — ${message}`);
    }
  }

  if (!chunks.length) {
    throw new Error("לא הופק אף קטע — כל המסמכים נכשלו בחילוץ");
  }

  // ---- 4: embed ------------------------------------------------------------
  const model = EMBED_MODELS[EMBED_TRANSPORT];
  log(`\nסה"כ ${chunks.length} קטעים. מחשב embeddings דרך ${model.modelId} (${EMBED_TRANSPORT})...`);

  // Reuse is keyed on the text, not the id: an instruction that was re-published
  // shifts its chunk numbering, so matching on id alone would silently reuse a
  // vector belonging to different words.
  const reusable = new Map<string, number[]>();
  if (options.reuseExisting !== false) {
    // Re-attach generated context BEFORE fingerprinting: the fingerprint covers
    // embeddingView, which includes the context, so attaching afterwards would
    // reuse vectors built without it and leave a silently stale index.
    const contexts = loadContexts(collection);
    let attached = 0;
    for (const c of chunks) {
      const ctx = contexts[c.id];
      if (ctx) { c.context = ctx; attached++; }
    }
    if (Object.keys(contexts).length) {
      console.log(`   הקשרים שצורפו: ${attached}/${chunks.length} (בקובץ: ${Object.keys(contexts).length})`);
    }

    const existing = await safeLoadExisting(collection);
    for (const c of existing) if (c.embedding) reusable.set(fingerprint(c), c.embedding);
    if (reusable.size) log(`   ${reusable.size} קטעים קיימים מריצה קודמת — לא מחושבים מחדש`);
  }

  const pending: number[] = [];
  let reused = 0;
  chunks.forEach((chunk, i) => {
    const hit = reusable.get(fingerprint(chunk));
    if (hit) {
      chunk.embedding = hit;
      reused++;
    } else {
      pending.push(i);
    }
  });

  let tokens = 0;
  if (pending.length) {
    // embeddingView, not .text — the passage is embedded together with its
    // document's title and official keywords. See embeddingView() below.
    const texts = pending.map(i => embeddingView(chunks[i]));
    const result = await embedAll(texts, "document", (done, total) => {
      if (done === total || done % (EMBED_BATCH * 2) === 0) {
        const mins = ((Date.now() - started) / 60000).toFixed(1);
        log(`   ${done}/${total}  (${mins} דק')`);
      }
    });

    result.vectors.forEach((vector, n) => {
      chunks[pending[n]].embedding = vector;
    });
    tokens = result.tokens || texts.reduce((sum, t) => sum + estimateTokens(t), 0);
  } else {
    log("   כל הקטעים כבר מוטמעים — אין קריאות בתשלום");
  }

  // ---- 5: merge into the collection index ---------------------------------
  await mergeIntoIndex(collection, chunks);

  const costUSD = (tokens / 1_000_000) * EMBED_PRICE_PER_1M;
  return {
    collection,
    documents,
    chunks: chunks.length,
    embedded: pending.length,
    reused,
    tokens,
    costUSD,
    elapsedMs: Date.now() - started,
    failures,
  };
}

/**
 * Chunks are matched for reuse by (id + text). Two different runs that produce
 * byte-identical text for the same id genuinely describe the same passage.
 */
function fingerprint(chunk: Chunk): string {
  // The fingerprint must cover EXACTLY what was sent to the embedding model,
  // which is embeddingView(chunk) and not chunk.text. Keying on the raw text
  // alone would let a re-run reuse a vector built before the header existed —
  // a silently stale index that passes every check it has.
  return `${chunk.id}\u0000${embeddingView(chunk)}`;
}

/**
 * What actually gets embedded: the passage, preceded by the identity of the
 * document it came from.
 *
 * A chunk from the middle of an instruction is a few paragraphs of legal prose
 * that never restate their own subject — the document said "פתיחת ספק" once, in
 * its title, and then talked about approvals for eight pages. Embedded bare,
 * that passage holds nothing pointing at the question a person actually asks.
 *
 * Measured 02/09/2026 against the live index: "איך פותחים ספק חדש במערכת?"
 * scored 0.416 against a 0.49 threshold and was refused — while its top hit was
 * instruction 1.7.2, which is exactly the right document. The answer was in the
 * corpus; the passage simply never said so in its own words.
 *
 * The official keywords go in for the same reason and are the stronger half of
 * the two: they are the ministry's formal vocabulary for the document, which is
 * precisely the wording a citizen does NOT use and therefore precisely what has
 * to be bridged. They were already being handed to the model in answer.ts; now
 * they also reach the vector, where they decide whether the passage is found at
 * all rather than only how it reads once it has been.
 *
 * `chunk.text` is deliberately left untouched. It is what the model quotes
 * from, and a synthetic header inside the quotable text would surface in
 * answers as words the instruction never wrote.
 */
export function embeddingView(chunk: Chunk): string {
  const parts: string[] = [];
  // The generated context sentence goes FIRST, before title and keywords: it is
  // the most specific statement of what this passage is about, and an embedding
  // weights early tokens most. Behind CONTEXTUAL_RETRIEVAL because turning it on
  // changes every vector and therefore requires a full re-embed.
  if (CONTEXTUAL_RETRIEVAL && chunk.context) parts.push(chunk.context);
  // A title only earns its place when it says something the code does not:
  // titleFor() falls back to the code itself for an untitled document.
  if (chunk.title && chunk.title !== chunk.code) parts.push(chunk.title);
  if (chunk.keywords?.length) parts.push(chunk.keywords.join(", "));
  if (!parts.length) return chunk.text;
  return `${parts.join(" | ")}

${chunk.text}`;
}

/**
 * Load the generated context sentences for a collection, if any.
 *
 * They live in a SIDECAR keyed by chunk id — `data/<collection>.context.json` —
 * and not inside the index, and that is the whole point. `ingest` rebuilds its
 * chunk list by re-extracting the source documents every run, so anything
 * written onto the index chunks is discarded the next time it runs. A sidecar
 * survives, and re-attaches by id.
 *
 * Chunk ids are `${code}#${index}` and stay stable while the source document
 * and the chunker do; a chunk whose id no longer exists simply finds no entry,
 * which is the correct outcome — a context sentence written for different text
 * must not be re-attached to new text.
 */
function loadContexts(collection: string): Record<string, string> {
  const path = join(LOCAL_DATA_DIR, `${collection}.context.json`);
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8")) as Record<string, string>;
  } catch {
    console.warn(`   אזהרה: ${path} אינו קריא — ההקשרים לא יצורפו`);
    return {};
  }
}

/** A missing index on the very first run is the normal case, not a failure. */
async function safeLoadExisting(collection: string): Promise<Chunk[]> {
  try {
    const { loadIndex } = await import("./store.js");
    const index = await loadIndex(collection);
    // Vectors from a different embedding model are not comparable, so they are
    // not reusable either — a transport switch means paying for a re-embed.
    if (index.embedTransport !== EMBED_TRANSPORT) return [];
    /**
     * A context sentence changes what was sent to the embedding model just as
     * surely as a change of model does, so it invalidates reuse the same way.
     * Comparing embeddingView alone cannot catch it: gen-context writes its
     * sentences onto the stored chunks, so both sides of the comparison carry
     * the context while only one side of the VECTOR ever saw it.
     */
    if (Boolean(index.contextual) !== CONTEXTUAL_RETRIEVAL) return [];
    return index.chunks;
  } catch {
    return [];
  }
}
