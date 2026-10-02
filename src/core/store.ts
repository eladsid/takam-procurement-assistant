/**
 * store.ts — the storage seam. (AD-3, AD-4, AD-6)
 *
 * One idea, and everything else here is a consequence of it: the core logic
 * must not know WHERE the data lives. `ingest`, `retrieve` and the Lambda all
 * say "give me the index for collection X" and this module decides whether that
 * means an S3 GetObject in us-east-1 or a `readFile` on the laptop. The switch
 * is a single config value (RUN_MODE), which is what makes the demo survive a
 * dead conference-room network without becoming a mock: it is the same code
 * path, the same JSON shape, the same arithmetic — only the disk changes.
 *
 * Two decisions in here are worth being able to defend out loud:
 *
 *  1. ONE INDEX PER COLLECTION (AD-4). `index/takam.json` and
 *     `index/aws-bedrock.json` are separate files. Answering a question about
 *     TAKAM never loads (or pays to scan) 2,400 pages of AWS documentation, and
 *     re-ingesting one corpus cannot corrupt another. The old code had a single
 *     `vectors.json` and a single module-level cache variable, which made a
 *     second collection literally impossible to add.
 *
 *  2. MERGE, NEVER OVERWRITE (AD-3). Writing an index is a read-modify-write
 *     keyed by chunk id, so dropping one new .docx into S3 adds its chunks and
 *     leaves the other 415 instructions exactly where they were. Overwriting
 *     the whole file would mean a single-document ingest silently deletes the
 *     corpus — a data-loss bug that only shows up in production, at the worst
 *     possible moment.
 *
 *  3. TWO STORAGE FORMATS, ONE CHOSEN BY SIZE (FR-4.3 / D-1). Small collections
 *     stay as one self-contained JSON file, exactly as before. Large ones split
 *     into `{collection}.json` (metadata) + `{collection}.vec` (the raw floats).
 *     Nothing outside this file knows which one it got — `loadIndex` returns the
 *     same `VectorIndex` shape either way. See the "Binary vector format"
 *     section below for the format and the reasoning.
 */

import { existsSync, mkdirSync } from "node:fs";
import { readFile, writeFile, readdir } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  ListObjectsV2Command,
} from "@aws-sdk/client-s3";
import { CONTEXTUAL_RETRIEVAL,
  RUN_MODE,
  S3_BUCKET,
  S3_REGION,
  DOCS_PREFIX,
  LEGACY_CORPUS_PREFIX,
  DEFAULT_COLLECTION,
  LEGACY_COLLECTION,
  TAKAM_CHAPTERS,
  LOCAL_DATA_DIR,
  CORPUS_DIR,
  EMBED_TRANSPORT,
  EMBED_MODELS,
  EMBED_REGION,
  VECTOR_BINARY_THRESHOLD,
  docsPrefixFor,
  indexKeyFor,
} from "./config.js";
import type { Chunk, VectorIndex } from "./types.js";

// ---------------------------------------------------------------------------
// Clients and paths
// ---------------------------------------------------------------------------

/**
 * The S3 client is created on first use rather than at import time. In local
 * mode it is never created at all, so `RUN_MODE=local` works on a machine with
 * no AWS credentials configured — which is the whole point of local mode.
 */
let s3: S3Client | null = null;
const client = (): S3Client => (s3 ??= new S3Client({ region: S3_REGION }));

/**
 * Local paths are resolved against the poc/ folder, never against the current
 * working directory. `npm run ingest` from poc/ and `tsx src/core/store.ts`
 * from the repo root must reach the same files; cwd-relative paths would not.
 */
const POC_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const localIndexPath = (collection: string): string =>
  resolve(POC_ROOT, LOCAL_DATA_DIR, `${collection}.json`);
const localCorpusDir = (): string => resolve(POC_ROOT, CORPUS_DIR);

/**
 * The companion vector file always sits beside its JSON, same base name, `.vec`
 * instead of `.json`. Both are derived from the SAME source — the existing
 * `indexKeyFor` / `localIndexPath` — rather than rebuilt from the prefix, so a
 * change to INDEX_PREFIX or LOCAL_DATA_DIR can never move one file and leave
 * the other behind.
 */
const vectorKeyFor = (collection: string): string =>
  indexKeyFor(collection).replace(/\.json$/, ".vec");
const localVectorPath = (collection: string): string =>
  localIndexPath(collection).replace(/\.json$/, ".vec");

/** Extensions we can actually turn into text. Everything else is skipped. */
/**
 * What counts as an ingestible document during DISCOVERY.
 *
 * This list must stay in step with the formats extract.ts can actually read;
 * when it falls behind, the effect is silent. On 02/09/2026 the corpus grew to
 * hold 51 PDFs, 92 spreadsheets, a deck and five legacy Office files — 148
 * documents — while this array still named four extensions. Any discovery-based
 * ingest would have stopped at 808 of 957 files with no error and no log line:
 * the annexes would simply never have existed as far as search was concerned.
 */
const DOCUMENT_EXTENSIONS = [
  ".docx", ".pdf", ".xlsx", ".pptx", ".doc", ".xlsb",
  ".md", ".txt", ".html",
];
const isDocument = (key: string): boolean =>
  DOCUMENT_EXTENSIONS.some(ext => key.toLowerCase().endsWith(ext));

// ---------------------------------------------------------------------------
// Small S3 helpers
// ---------------------------------------------------------------------------

/**
 * S3 hands back a stream, not a Buffer. Same pattern as the original
 * ingest.ts — collect the pieces, then concatenate once.
 */
async function s3GetBytes(key: string): Promise<Buffer> {
  const res = await client().send(new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }));
  const parts: Buffer[] = [];
  for await (const c of res.Body as AsyncIterable<Uint8Array>) parts.push(Buffer.from(c));
  return Buffer.concat(parts);
}

/**
 * ListObjectsV2 returns at most 1,000 keys per call and sets IsTruncated when
 * there are more. The AWS documentation corpus is ~2,400 pages, so a single
 * un-paginated call would silently ingest the first 1,000 and quietly lose the
 * rest — the kind of bug that looks like "the model doesn't know that page"
 * rather than like a bug.
 */
async function s3ListKeys(prefix: string): Promise<string[]> {
  const keys: string[] = [];
  let token: string | undefined;

  do {
    const res = await client().send(new ListObjectsV2Command({
      Bucket: S3_BUCKET,
      Prefix: prefix,
      ContinuationToken: token,
    }));
    for (const obj of res.Contents ?? []) {
      // A "folder" placeholder object ends with "/" and has no content.
      if (obj.Key && !obj.Key.endsWith("/")) keys.push(obj.Key);
    }
    token = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (token);

  return keys.sort();
}

// ---------------------------------------------------------------------------
// Binary vector format (FR-4.3 / D-1)
// ---------------------------------------------------------------------------

/**
 * WHY THIS EXISTS AT ALL — the measurement, not a hunch.
 *
 * TAKAM is 545 chunks and 7.5MB of JSON. That is fine, and it stays JSON.
 * The pilot over the Bedrock user guide projects 1,092 pages → ~6,534 chunks →
 * ~90MB of JSON, and almost all of that 90MB is 6.7 million floating-point
 * numbers written out as decimal TEXT ("0.023445871", eleven-plus characters
 * for four bytes of information). Every cold Lambda start would have to read
 * 90MB and then run 6.7 million string→number conversions before it can answer
 * the first question. The identical vectors stored as raw little-endian
 * Float32 are ~27MB and need no conversion whatsoever: the bytes on disk are
 * already the bytes the CPU does arithmetic on.
 *
 * THE FORMAT. Two files per collection:
 *
 *   index/{collection}.json  the full VectorIndex envelope and every chunk's
 *                            metadata, with `embedding` OMITTED from each
 *                            chunk, plus two extra fields:
 *                              format:     "binary-f32"
 *                              vectorFile: "{collection}.vec"
 *   index/{collection}.vec   every vector, concatenated, little-endian Float32,
 *                            in exactly the order of the JSON `chunks` array.
 *
 * WHY THE .vec FILE HAS NO HEADER. A header would repeat information the JSON
 * already carries — `dims` and `chunks.length` — and two copies of the same
 * fact are two facts that can disagree. Worse, a header would let the file look
 * self-describing while still being wrong, which invites trusting it. Here the
 * JSON is the single source of truth about the shape, the .vec is pure payload,
 * and the two are cross-checked on every load by the one arithmetic identity
 * that must hold: byteLength === chunks × dims × 4. That check is the whole
 * safety story (see attachVectors), and it is a stronger guarantee than a magic
 * number would have given.
 *
 * MEASURED on the real TAKAM index (545 chunks × 1024 dims), same data both
 * ways: 7,830,889 bytes as one JSON file → 884,349 bytes of JSON metadata plus
 * 2,232,320 bytes of .vec = 3,116,669 bytes, i.e. 39.8%. The vectors themselves
 * shrink 7.0MB → 2.13MB (3.3×); what is left is the chunk text, which JSON
 * stores perfectly well and which this format does not touch.
 *
 * WHY NOT ONE FILE WITH THE BINARY APPENDED. Because then it is neither: you
 * cannot `cat` it, S3 cannot serve a byte range of the metadata alone, and any
 * tool that expects `.json` to be JSON breaks. Two files, each honest about
 * what it is.
 */
const BINARY_FORMAT = "binary-f32" as const;

/** Bytes per Float32. Named because it appears in the load-time size check. */
const F32_BYTES = 4;

/**
 * The on-disk envelope. `VectorIndex` (types.ts) is the shared in-memory
 * contract used by six other modules and is deliberately NOT changed for a
 * storage detail — these two fields exist only between `saveIndex` and
 * `loadIndex` and never escape this file. An old JSON index simply has neither,
 * which is exactly what makes the change backward compatible: absent marker =
 * the format that already worked.
 */
interface StoredIndex extends VectorIndex {
  format?: typeof BINARY_FORMAT;
  /** Bare file name, e.g. "takam.vec" — never a path or an S3 key. */
  vectorFile?: string;
}

/**
 * `INDEX_FORMAT` — force the choice instead of letting size decide.
 *
 *   auto (default)  binary when the collection is large enough, JSON otherwise
 *   json            always JSON, whatever the size
 *   binary          always binary (still refuses on a partially-embedded index)
 *
 * It is read here rather than in config.ts because it is a property of the
 * storage layer and nothing outside this file may act on it — a caller that
 * could see the format would be a caller that could start depending on it.
 * Its real job is testability: without it, the binary path could only ever be
 * exercised by building a 2,000-chunk corpus, which is a slow and expensive way
 * to test a file writer.
 */
const INDEX_FORMAT = (process.env.INDEX_FORMAT ?? "auto") as "auto" | "json" | "binary";

/**
 * Float32Array reads and writes in the HOST's byte order, and the format is
 * specified as little-endian. Every machine this runs on — an x86-64 laptop, an
 * x86-64 or Graviton Lambda — is little-endian, so the fast path is correct and
 * costs nothing. But "correct on every machine we happened to try" is how you
 * get a corrupt index on the one machine you did not, and a byte-swapped vector
 * does not crash: it produces confident, meaningless similarity scores. So the
 * assumption is checked once, at load/save time, and stated out loud.
 */
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function assertLittleEndian(): void {
  if (!IS_LITTLE_ENDIAN) {
    throw new Error(
      "פורמט הווקטורים הבינארי מוגדר כ-little-endian, והמכונה הזאת היא big-endian. " +
      "קריאה ישירה הייתה מחזירה מספרים שגויים בלי לקרוס — יש להשתמש ב-INDEX_FORMAT=json על מכונה כזאת.",
    );
  }
}

/**
 * Is every chunk carrying a usable vector of the right length?
 *
 * This gates the binary format, and the reason is structural rather than
 * defensive: in the .vec file a chunk's position IS its identity — chunk i owns
 * bytes [i·dims·4, (i+1)·dims·4). There is no way to write "this one has no
 * vector". The alternatives were to pad the gap with zeros (a zero vector
 * scores NaN in cosine and would quietly poison ranking) or to drop the chunk
 * (silent data loss). Both are worse than simply staying on the JSON format,
 * which represents a missing embedding natively, for the one case where an
 * ingest did not finish. So: a partially-embedded index is always JSON.
 */
function fullyEmbedded(chunks: Chunk[], dims: number): boolean {
  return dims > 0 && chunks.every(c => c.embedding && c.embedding.length === dims);
}

/**
 * Which format should this write use? JSON is the answer whenever there is any
 * doubt — that is the backward-compatibility rule stated as code.
 */
function chooseFormat(index: VectorIndex): "json" | typeof BINARY_FORMAT {
  if (INDEX_FORMAT === "json") return "json";

  const wantsBinary =
    INDEX_FORMAT === "binary" || index.chunks.length >= VECTOR_BINARY_THRESHOLD;
  if (!wantsBinary) return "json";

  if (!fullyEmbedded(index.chunks, index.dims)) {
    // Loud, because this silently changes the storage layout of a large index
    // and the operator should know why their 90MB collection is still JSON.
    console.warn(
      `   אזהרה: האוסף "${index.collection}" אינו מוטמע במלואו (חלק מהקטעים ללא וקטור, ` +
      `או באורך שונה מ-${index.dims}) — נשמר בפורמט JSON ולא בפורמט הבינארי. ` +
      `יש להשלים ingest ולשמור שוב.`,
    );
    return "json";
  }

  assertLittleEndian();
  return BINARY_FORMAT;
}

/**
 * Pack every chunk's vector into one contiguous little-endian Float32 buffer.
 *
 * `Float32Array.set` accepts both a plain `number[]` (what a fresh ingest
 * produces) and another Float32Array (what a binary-loaded index hands back on
 * a re-save), so one loop covers the round-trip in both directions. Each
 * `number` is a 64-bit double being narrowed to 32 bits here; that is the
 * deliberate trade of this whole format — see loadVectors for why the precision
 * loss does not matter for cosine similarity.
 */
function packVectors(chunks: Chunk[], dims: number): Buffer {
  const packed = new Float32Array(chunks.length * dims);

  chunks.forEach((chunk, i) => {
    const vector = chunk.embedding as ArrayLike<number> | undefined;
    // chooseFormat already guaranteed this; re-checked because writing a short
    // vector here is exactly the silent corruption the format must not allow.
    if (!vector || vector.length !== dims) {
      throw new Error(
        `לא ניתן לכתוב את קובץ הווקטורים: לקטע "${chunk.id}" יש ${vector?.length ?? 0} מימדים ` +
        `במקום ${dims}. כתיבה כזאת הייתה מזיזה את כל הווקטורים שאחריו ומחזירה תוצאות שגויות בשקט.`,
      );
    }
    packed.set(vector, i * dims);
  });

  // A Buffer VIEW over the same memory — no copy of 27MB just to write it out.
  return Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength);
}

/**
 * Turn the .vec bytes back into one vector per chunk, in place.
 *
 * Each chunk gets a `subarray` — a WINDOW onto the single buffer, not a copy.
 * The whole index therefore costs one 27MB allocation rather than 6,534
 * separate arrays, and rehydration is O(chunks) pointer arithmetic instead of
 * O(floats) parsing.
 *
 * THE ONE PIECE OF SLEIGHT OF HAND, stated plainly because a reviewer will
 * (rightly) stop on it: `Chunk.embedding` is typed `number[]` and a
 * Float32Array is not literally that, hence the cast. It is safe because of
 * what the consumers actually do — `retrieve.ts` checks `.length` and `cosine`
 * reads `a[i]`, and a Float32Array answers both identically. The alternative,
 * `Array.from(...)`, would convert all 6.7 million floats back into 64-bit
 * doubles: it would give up the memory win (~53MB instead of ~27MB) and re-add
 * a per-float loop, i.e. give back most of what this format was built to save.
 * Widening the shared type in types.ts was the other option and was rejected as
 * out of scope: it would push a storage detail into six unrelated modules.
 * The cast is confined to this one line, and the round-trip test proves it.
 *
 * ON PRECISION — this format IS lossy, and here is the measurement rather than
 * a reassurance. Bedrock returns each component as a decimal with more digits
 * than 32 bits can hold, so JSON.parse produces a 64-bit double and writing it
 * as Float32 rounds. Measured over all 545 × 1024 = 558,080 components of the
 * TAKAM index, the largest error was 6.25e-9 — half of one float32 ulp, i.e.
 * the theoretical minimum, so nothing is going wrong beyond the narrowing
 * itself. The number that matters is the downstream one: the cosine scores that
 * come out are identical to the JSON path to seven decimal places (0.7563455
 * either way), and the ranking is unchanged. Retrieval compares vectors that
 * differ from each other in the second decimal place; an error in the ninth
 * cannot reorder them. Storing 64-bit doubles to protect digits the model never
 * meant is what would actually be the mistake.
 */
function attachVectors(chunks: Chunk[], bytes: Buffer, dims: number, where: string): void {
  assertLittleEndian();

  const expected = chunks.length * dims * F32_BYTES;
  if (bytes.byteLength !== expected) {
    throw new Error(
      `אי-התאמה בגודל קובץ הווקטורים ${where}: צפויים ${expected.toLocaleString()} בתים ` +
      `(${chunks.length.toLocaleString()} קטעים × ${dims} מימדים × ${F32_BYTES} בתים) ` +
      `והתקבלו ${bytes.byteLength.toLocaleString()}. קובץ חתוך או שאינו תואם לקובץ ה-JSON היה ` +
      `מחזיר וקטורים מוזזים — כלומר תשובות סבירות-למראה ושגויות לחלוטין — ולכן הטעינה נעצרת כאן. ` +
      `יש להריץ ingest מחדש לאוסף הזה.`,
    );
  }

  /**
   * A Float32Array view requires its byte offset to be a multiple of 4. Node
   * hands back Buffers that are windows into a shared pool, so `byteOffset` is
   * NOT reliably 0 — and an unaligned view throws RangeError rather than
   * returning wrong data. Copying in that case costs one allocation and happens
   * rarely; guessing would cost a crash on some file sizes and not others.
   */
  const aligned = bytes.byteOffset % F32_BYTES === 0
    ? bytes
    : Buffer.from(bytes);

  const all = new Float32Array(aligned.buffer, aligned.byteOffset, chunks.length * dims);

  chunks.forEach((chunk, i) => {
    chunk.embedding = all.subarray(i * dims, (i + 1) * dims) as unknown as number[];
  });
}

/**
 * Prepare an index for `JSON.stringify`.
 *
 * Non-obvious and load-bearing: `JSON.stringify(new Float32Array([1,2]))` does
 * NOT produce `[1,2]` — it produces `{"0":1,"1":2}`, because a typed array is
 * an object, not an Array. So an index that was LOADED from the binary format
 * and is then SAVED as JSON (dropped below the threshold, or INDEX_FORMAT=json)
 * would write an index that parses fine and whose every vector is an unusable
 * object. Converting back to a plain array here is what keeps the two formats
 * genuinely interchangeable in both directions. The copy only happens for
 * chunks that actually hold a typed array, so the ordinary JSON path allocates
 * nothing extra.
 */
function toJsonSafe(index: VectorIndex): VectorIndex {
  return {
    ...index,
    chunks: index.chunks.map(chunk => {
      const vector = chunk.embedding;
      if (!vector || Array.isArray(vector)) return chunk;
      return { ...chunk, embedding: Array.from(vector as ArrayLike<number>) };
    }),
  };
}

// ---------------------------------------------------------------------------
// Index cache
// ---------------------------------------------------------------------------

/**
 * One cache entry PER COLLECTION, not one global singleton.
 *
 * Why a Map and not a single variable: the old `let cache: VectorStore | null`
 * could only ever hold one corpus, so asking a TAKAM question and then an AWS
 * question would have served the second from the first one's vectors. That
 * single line was the actual multi-collection blocker.
 *
 * Why cache at all: in a warm Lambda the container survives between requests.
 * A cold call downloads a multi-megabyte JSON from S3 and parses it (~3s); a
 * warm call reads this Map (~300ms end to end). Same code, ten times faster,
 * for the price of one Map — and it is also why an index that changed in S3 is
 * only picked up by a NEW container, which is exactly why every write here
 * refreshes the cache instead of just invalidating it.
 */
const indexCache = new Map<string, VectorIndex>();

/** Drop cached indexes. Useful in tests and after an out-of-band S3 upload. */
export function clearIndexCache(collection?: string): void {
  if (collection) indexCache.delete(collection);
  else indexCache.clear();
}

/**
 * Read one collection's index. S3 in aws mode, local disk in local mode.
 * Throws a Hebrew error when the index does not exist yet — the fix is always
 * "run the ingest", so the message says so instead of leaking a NoSuchKey.
 *
 * Both storage formats come out of here as the same `VectorIndex`. Format
 * detection is one field: if the JSON says `format: "binary-f32"` the companion
 * .vec is read and its floats are attached to the chunks; if the field is
 * ABSENT — which is every index written before this feature existed — the
 * embeddings are parsed straight out of the JSON exactly as they always were.
 * "Absent means the old way" is what makes an existing data/takam.json keep
 * working with no migration step and no flag to remember to set.
 */
export async function loadIndex(collection: string): Promise<VectorIndex> {
  const cached = indexCache.get(collection);
  if (cached) return cached;

  let raw: string;

  if (RUN_MODE === "aws") {
    const key = indexKeyFor(collection);
    try {
      raw = (await s3GetBytes(key)).toString("utf-8");
    } catch (err: any) {
      if (err?.name === "NoSuchKey" || err?.$metadata?.httpStatusCode === 404) {
        throw new Error(
          `לא נמצא אינדקס לאוסף "${collection}" ב-s3://${S3_BUCKET}/${key} — יש להריץ קודם ingest לאוסף הזה.`,
        );
      }
      throw err;
    }
  } else {
    const file = localIndexPath(collection);
    if (!existsSync(file)) {
      throw new Error(
        `לא נמצא אינדקס לאוסף "${collection}" בנתיב ${file} — יש להריץ קודם ingest לאוסף הזה.`,
      );
    }
    raw = await readFile(file, "utf-8");
  }

  const index = JSON.parse(raw) as StoredIndex;

  if (index.format === BINARY_FORMAT) {
    const bytes = await readVectorFile(collection);
    attachVectors(index.chunks, bytes, index.dims, vectorLocation(collection));
    // The marker describes the file on disk, not the object in memory: by this
    // point the vectors are attached and this object is indistinguishable from
    // a JSON-loaded one. Leaving the field set would let a later `saveIndex`
    // believe a decision it is supposed to make fresh, every time.
    delete index.format;
    delete index.vectorFile;
  }

  indexCache.set(collection, index);
  return index;
}

/** Human-readable location of the .vec, for error messages. */
const vectorLocation = (collection: string): string =>
  RUN_MODE === "aws"
    ? `s3://${S3_BUCKET}/${vectorKeyFor(collection)}`
    : localVectorPath(collection);

/**
 * Fetch the companion vector file. A missing .vec next to a JSON that claims
 * the binary format is a hard, loud failure and never a fallback: the metadata
 * is intact, so the system could happily serve an index in which every chunk
 * has no vector and retrieval silently returns nothing relevant. Better to
 * refuse and name the fix.
 */
async function readVectorFile(collection: string): Promise<Buffer> {
  const missing = new Error(
    `לא נמצא קובץ הווקטורים ${vectorLocation(collection)} — קובץ ה-JSON של האוסף "${collection}" ` +
    `מסומן כפורמט בינארי (${BINARY_FORMAT}) אך הקובץ הנלווה שמכיל את הווקטורים עצמם חסר. ` +
    `שני הקבצים חייבים לנוע יחד. יש להריץ ingest מחדש לאוסף הזה.`,
  );

  if (RUN_MODE === "aws") {
    try {
      return await s3GetBytes(vectorKeyFor(collection));
    } catch (err: any) {
      if (err?.name === "NoSuchKey" || err?.$metadata?.httpStatusCode === 404) throw missing;
      throw err;
    }
  }

  const file = localVectorPath(collection);
  if (!existsSync(file)) throw missing;
  return await readFile(file);
}

/**
 * Write one collection's index back to wherever it belongs, and refresh the
 * cache so the process that just wrote it does not keep serving the old copy.
 *
 * The envelope (model, transport, region, dims, timestamp) is stamped HERE, on
 * every single write, rather than trusted from the caller. Vectors produced by
 * two different embedding models are not comparable — mixing them does not
 * throw, it just returns quietly wrong answers — so the file must always be
 * able to say which model produced it. Centralising the stamp means no caller
 * can forget it.
 *
 * The storage FORMAT is decided here too, and for the same reason: it is a
 * property of the file, not of the caller. `ingest`, the Lambda and the CLI all
 * just say "save this"; whether that becomes one JSON file or a JSON + .vec
 * pair is this function's business alone. (FR-4.3 / D-1)
 */
export async function saveIndex(index: VectorIndex): Promise<void> {
  const model = EMBED_MODELS[EMBED_TRANSPORT];

  const stamped: VectorIndex = {
    ...index,
    embedModel: model.modelId,
    embedTransport: EMBED_TRANSPORT,
    contextual: CONTEXTUAL_RETRIEVAL,
    region: EMBED_REGION,
    dims: model.dims,
    indexedAt: new Date().toISOString(),   // surfaced under every answer: the corpus has a date
  };

  if (chooseFormat(stamped) === BINARY_FORMAT) {
    await saveBinary(stamped);
  } else {
    await saveJson(stamped);
  }

  indexCache.set(stamped.collection, stamped);
}

/** The original single-file layout. Unchanged behaviour, now just one branch. */
async function saveJson(index: VectorIndex): Promise<void> {
  // toJsonSafe matters only on the binary→JSON direction; see its comment.
  const body = JSON.stringify(toJsonSafe(index));

  if (RUN_MODE === "aws") {
    await client().send(new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: indexKeyFor(index.collection),
      Body: body,
      ContentType: "application/json",
    }));
    return;
  }

  const file = localIndexPath(index.collection);
  mkdirSync(dirname(file), { recursive: true });
  await writeFile(file, body, "utf-8");
}

/**
 * The split layout: metadata as JSON, vectors as raw Float32.
 *
 * WRITE ORDER IS DELIBERATE — .vec first, .json second — and it is the only
 * crash-safety this design has. The JSON is the pointer: nothing reads the .vec
 * unless a JSON tells it to. So if the process dies between the two writes, the
 * OLD json is still on disk, still self-consistent, still loadable, and the
 * half-written .vec is an orphan nobody looks at. Written in the other order, a
 * crash would leave a JSON promising a .vec that is stale or absent — a broken
 * index produced by an interrupted save, which is the worst outcome available.
 *
 * Neither store is transactional across two objects, so a torn pair is still
 * possible in principle (a truncated .vec that finished writing its header but
 * not its tail). That is precisely what the byte-length check in `attachVectors`
 * exists to catch on the way back in.
 *
 * Known, accepted leftover: a collection that used to be binary and is saved as
 * JSON leaves its old .vec behind, orphaned. It is harmless — no JSON points at
 * it any more — and deleting it would mean requiring s3:DeleteObject in the IAM
 * policy, i.e. granting the ingest role the ability to destroy data, to reclaim
 * some bytes. Not a trade worth making here.
 */
async function saveBinary(index: VectorIndex): Promise<void> {
  const vectors = packVectors(index.chunks, index.dims);

  // The JSON twin: same envelope, same chunks, minus the vectors, plus the two
  // fields that tell `loadIndex` where the vectors went.
  const envelope: StoredIndex = {
    ...index,
    format: BINARY_FORMAT,
    vectorFile: basename(vectorKeyFor(index.collection)),
    chunks: index.chunks.map(({ embedding, ...rest }) => rest),
  };
  const body = JSON.stringify(envelope);

  if (RUN_MODE === "aws") {
    await client().send(new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: vectorKeyFor(index.collection),
      Body: vectors,
      ContentType: "application/octet-stream",
    }));
    await client().send(new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: indexKeyFor(index.collection),
      Body: body,
      ContentType: "application/json",
    }));
    return;
  }

  const file = localIndexPath(index.collection);
  mkdirSync(dirname(file), { recursive: true });
  await writeFile(localVectorPath(index.collection), vectors);
  await writeFile(file, body, "utf-8");
}

/**
 * Add or update chunks in a collection WITHOUT touching the ones that were
 * already there. (AD-3)
 *
 * The rule is merge-by-id:
 *   - a chunk id present only in the existing index  → survives untouched;
 *   - a chunk id present in both                     → replaced by the new one;
 *   - a chunk id present only in newChunks           → appended.
 *
 * This is what makes live ingest of a single document safe. The S3-event
 * Lambda fires with ONE new file; it re-chunks and re-embeds that file only,
 * calls this, and the other 415 instructions are still in the index afterwards.
 * With a wholesale overwrite, the very first live upload would have wiped the
 * corpus and the failure would have looked like "the assistant forgot
 * everything" rather than like the write bug it is.
 *
 * The chunk id (`${code}#${index}`) carries the document it came from, so
 * re-ingesting a document that got SHORTER leaves its now-orphaned tail chunks
 * behind. That is a deliberate trade: a stale extra passage ranks low and is
 * harmless, whereas an accidental delete is not recoverable. A full rebuild
 * (ingest of the whole collection into a fresh index) is the way to prune.
 */
export async function mergeIntoIndex(
  collection: string,
  newChunks: Chunk[],
): Promise<VectorIndex> {
  const existing = (await indexExists(collection))
    ? (await loadIndex(collection)).chunks
    : [];

  // Insertion order = existing first, then genuinely new chunks. A Map keyed by
  // id gives last-write-wins semantics in one pass, without an O(n²) scan.
  const byId = new Map<string, Chunk>();
  for (const c of existing) byId.set(c.id, c);
  for (const c of newChunks) byId.set(c.id, c);

  const model = EMBED_MODELS[EMBED_TRANSPORT];

  // saveIndex re-stamps the envelope; these values just keep the object valid
  // in the meantime and are what the caller gets back.
  const merged: VectorIndex = {
    collection,
    embedModel: model.modelId,
    embedTransport: EMBED_TRANSPORT,
    contextual: CONTEXTUAL_RETRIEVAL,
    region: EMBED_REGION,
    dims: model.dims,
    indexedAt: new Date().toISOString(),
    chunks: [...byId.values()],
  };

  await saveIndex(merged);
  return merged;
}

/** Cheap existence check — used to decide between "merge" and "create". */
export async function indexExists(collection: string): Promise<boolean> {
  if (indexCache.has(collection)) return true;

  if (RUN_MODE === "aws") {
    // A prefix listing is used rather than HeadObject on purpose: listing needs
    // only s3:ListBucket, so the check still works under a tight IAM policy.
    const keys = await s3ListKeys(indexKeyFor(collection));
    return keys.length > 0;
  }
  return existsSync(localIndexPath(collection));
}

/**
 * Which of a virtual collection's members actually have an index right now.
 *
 * This exists because the member list and the built indexes are deliberately
 * NOT the same set. `aws-all` names every AWS collection the config declares
 * (43 of them), while only the guides we chose to embed have an index (9). That
 * gap is the design — the archive on disk is larger than what the demo answers
 * from — so resolving the list before searching is not a defensive nicety: it
 * is the difference between 9 index loads per question and 42, with 33 of them
 * throwing and being caught.
 *
 * The result is cached per collection because it only changes when an ingest
 * runs, and an ingest restarts the process.
 */
const memberCache = new Map<string, string[]>();

export async function resolveMembers(def: { id: string; members?: string[] }): Promise<string[]> {
  if (!def.members?.length) return [];
  const cached = memberCache.get(def.id);
  if (cached) return cached;

  const flags = await Promise.all(def.members.map(indexExists));
  const present = def.members.filter((_, i) => flags[i]);
  memberCache.set(def.id, present);
  return present;
}

/**
 * Is this collection ready to be offered to a user?
 *
 * For a real collection that is exactly "does its index exist". For a VIRTUAL
 * one it is "do at least TWO of its members exist" — and the choice of two
 * rather than all deserves the explanation, because "all" was the first rule
 * written here and it was wrong.
 *
 * "All" assumed the member list describes what SHOULD exist. It does not: it
 * describes every AWS guide the config knows about, most of which were
 * downloaded but deliberately never embedded. Requiring all of them would mean
 * "search everything" never becomes available no matter how much is indexed —
 * a permanently hidden feature.
 *
 * Two, rather than one, because a "search across everything" that resolves to a
 * single corpus is not a cross-collection search; it is that corpus wearing a
 * misleading label, and the user would be better served by picking it directly.
 */
export async function collectionAvailable(def: { id: string; members?: string[] }): Promise<boolean> {
  if (!def.members?.length) return indexExists(def.id);
  return (await resolveMembers(def)).length >= 2;
}

/**
 * Every document key belonging to a collection, sorted.
 *
 * TRANSITIONAL COMPATIBILITY SHIM — read this before "cleaning it up":
 * the 18 chapter-7 TAKAM files were uploaded to `corpus/` back when there was
 * only one corpus and the per-collection `docs/{collection}/` layout did not
 * exist yet. Rather than move objects in S3 (a migration with its own failure
 * modes, run against the only copy of the data), the `takam` collection falls
 * back to the legacy prefix when `docs/takam/` is empty. New uploads go to the
 * new layout, both are readable, and the day `docs/takam/` has content the
 * fallback stops firing on its own. Delete this branch only after the legacy
 * objects are gone.
 */
export async function listDocumentKeys(collection: string): Promise<string[]> {
  if (RUN_MODE === "aws") {
    const keys = (await s3ListKeys(docsPrefixFor(collection))).filter(isDocument);
    if (keys.length > 0) return keys;

    // The legacy `corpus/` prefix belongs to the ORIGINAL demo collection. This
    // read `collection === DEFAULT_COLLECTION`, which meant "takam" when it was
    // written and silently came to mean "takam-all" when the default moved on
    // 02/09/2026 — pointing the shim at a virtual collection that has no
    // documents, while the real one it was written for stopped matching.
    if (collection === LEGACY_COLLECTION) {
      const legacy = (await s3ListKeys(LEGACY_CORPUS_PREFIX)).filter(isDocument);
      if (legacy.length > 0) return legacy;
    }
    return [];
  }

  /**
   * Local mode mirrors the same shape: a per-collection sub-folder when one
   * exists, otherwise the flat corpus folder.
   *
   * The flat fallback is now restricted to the legacy collection, and that
   * restriction is a safety fix rather than tidying. `corpus/` used to hold the
   * 18 demo files, so falling back to it was harmless. It now holds all 957
   * documents, which made `npm run ingest -- --collection=takam-7` quietly
   * ingest the ENTIRE corpus into chapter 7 — and because mergeIntoIndex never
   * deletes, the only way back was to delete the index file by hand.
   *
   * Chapter collections are built by ingest-takam-chapters.ts, which passes
   * explicit keys from the manifest and never reaches this function.
   */
  const perCollection = join(localCorpusDir(), collection);
  let dir: string;
  if (existsSync(perCollection)) dir = perCollection;
  else if (collection === LEGACY_COLLECTION) dir = localCorpusDir();
  else return [];
  if (!existsSync(dir)) return [];

  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter(e => e.isFile() && isDocument(e.name))
    .map(e => join(dir, e.name))
    .sort();
}

/**
 * Read one document by the key `listDocumentKeys` returned — an S3 key in aws
 * mode, a file path in local mode. Callers (mammoth for .docx, plain text for
 * .md) only ever see bytes plus a file name, so they never learn which it was.
 */
export async function readDocument(key: string): Promise<{ fileName: string; bytes: Buffer }> {
  if (RUN_MODE === "aws") {
    return { fileName: basename(key), bytes: await s3GetBytes(key) };
  }
  const file = resolve(POC_ROOT, key);
  return { fileName: basename(file), bytes: await readFile(file) };
}

/**
 * Read a non-document companion file that sits beside a collection's documents
 * — in practice the corpus manifest, which carries the official titles and the
 * catalogue keywords that the .docx files themselves do not contain.
 *
 * Returns null when absent, because absent is a legitimate state: an AWS
 * documentation collection has no manifest, and a missing one only degrades a
 * citation from "instruction 7.6.1 — Exemption from tender" to "7.6.1". A hard
 * failure here would block ingestion over a cosmetic gap.
 */
export async function readAuxFile(collection: string, name: string): Promise<Buffer | null> {
  try {
    if (RUN_MODE === "aws") {
      /**
       * Per-collection first, then the shared legacy location.
       *
       * Without the fallback this only ever looked at
       * `docs/takam-7/manifest-all.json`, while the manifest that describes the
       * whole corpus is uploaded once to the legacy prefix. Every chapter
       * ingest in the cloud therefore found nothing and indexed its documents
       * with no title and no keywords. The local branch below already had this
       * fallback; the cloud branch did not.
       */
      const own = await s3GetBytes(`${docsPrefixFor(collection)}${name}`).catch(() => null);
      if (own) return own;
      return await s3GetBytes(`${LEGACY_CORPUS_PREFIX}${name}`);
    }
    const candidates = [
      resolve(POC_ROOT, CORPUS_DIR, collection, name),
      resolve(POC_ROOT, CORPUS_DIR, name),
    ];
    for (const file of candidates) {
      if (existsSync(file)) return await readFile(file);
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Which collection does this key belong to?
 *
 * Used by the S3-event Lambda: the event says "docs/aws-bedrock/agents.md was
 * created" and the ingest needs the collection name to know which index to
 * merge into. Legacy `corpus/…` keys predate the layout and belong to the
 * default collection (takam) — same shim as in listDocumentKeys.
 */
export function collectionFromKey(key: string): string {
  if (key.startsWith(DOCS_PREFIX)) {
    const rest = key.slice(DOCS_PREFIX.length);
    const slash = rest.indexOf("/");
    if (slash > 0) return rest.slice(0, slash);
  }

  /**
   * A TAKAM document names its own chapter: "H.7.10.7.docx" belongs to chapter
   * 7, so it goes to `takam-7`.
   *
   * This used to return DEFAULT_COLLECTION, and after the default became the
   * VIRTUAL collection `takam-all` that was actively broken: the live-ingest
   * Lambda would build a real `index/takam-all.json` that no query ever opens,
   * because a virtual collection is answered by searching its members. The demo
   * would log a successful ingest for a document that had become unfindable —
   * the worst possible outcome in front of an audience, since nothing errors.
   *
   * Routing by chapter puts the document where the search will actually look.
   */
  const chapter = /(?:^|[/])H[.](\d+)[.]/.exec(key)?.[1];
  if (chapter && TAKAM_CHAPTERS.some(([c]) => c === chapter)) return `takam-${chapter}`;

  /**
   * Anything unrecognised goes to the legacy demo collection rather than
   * throwing: an ingest that files a document slightly wrong is recoverable,
   * one that crashes the event Lambda on every retry is noisier and no more
   * correct. It must never be a virtual collection, which cannot hold anything.
   */
  return LEGACY_COLLECTION;
}
