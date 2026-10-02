/**
 * extract.ts — bytes in, readable prose out.
 *
 * The POC ingests two corpora that have nothing in common except that both are
 * text: Hebrew TAKAM instructions published as Word files (.docx), and AWS
 * service guides published as Markdown (.md). Rather than write two ingest
 * scripts, the format difference is confined to this one file — everything
 * downstream (chunk, embed, retrieve, answer) sees a plain string and never
 * learns where it came from. Adding a third format later means adding one case
 * here, not touching the pipeline.
 *
 * The other job of this file is the CITATION HANDLE: the short code the user
 * sees under an answer ("הוראה 7.6.1"). It is derived from the file name
 * because the file name is the only identifier that both corpora actually
 * carry — and because a citation that cannot be traced back to a real document
 * is worse than no citation at all.
 */

import mammoth from "mammoth";
import JSZip from "jszip";
import { PDFParse } from "pdf-parse";
import { collectionDef } from "./config.js";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/**
 * The bare file name, with any S3 key prefix or local directory stripped.
 * Callers hand us "docs/takam/H.7.6.1.docx" from S3 and
 * "corpus\\H.7.6.1.docx" from Windows, so both separators are handled.
 */
function baseName(fileName: string): string {
  return fileName.split(/[\\/]/).pop() ?? fileName;
}

/** Lower-cased extension including the dot, or "" when there is none. */
function extensionOf(fileName: string): string {
  const name = baseName(fileName);
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/** File name without its extension. */
function stripExtension(fileName: string): string {
  const name = baseName(fileName);
  const dot = name.lastIndexOf(".");
  return dot === -1 ? name : name.slice(0, dot);
}

/**
 * Normalisation applied to every format before it leaves this module.
 *
 * - CRLF → LF, so the blank-line paragraph split in chunk.ts sees one shape.
 * - Invisible formatting characters removed — the BOM and the bidi direction
 *   marks (U+200E, U+200F, U+202A-U+202E, U+2066-U+2069) that Hebrew Word
 *   documents are peppered with. They render as nothing, but they are real
 *   characters: they consume the character budget, and they change the token
 *   stream the embedding model sees, which means two identical sentences can
 *   produce two different vectors purely because Word inserted a direction mark
 *   in one of them.
 * - Runs of 3+ blank lines collapsed to one blank line: the paragraph splitter
 *   only needs one, and Markdown-to-prose conversion leaves gaps behind.
 */
function normalise(text: string): string {
  // \p{Cf} is the Unicode "format" category: the BOM, the LRM/RLM direction
  // marks, the bidi embedding and isolate controls, and the soft hyphen — every
  // invisible character Word sprinkles through a Hebrew document. Matching the
  // category rather than listing the code points keeps the rule readable: a
  // literal invisible character in the source is impossible to see or diff.
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/\p{Cf}/gu, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ---------------------------------------------------------------------------
// Markdown → prose
// ---------------------------------------------------------------------------

/**
 * Strip Markdown syntax while keeping everything a reader (and a retriever)
 * would consider content.
 *
 * Why plain regex and not a Markdown parser: a parser would add a dependency
 * and an AST to reason about, in exchange for correctness on constructs the AWS
 * docs do not use (nested reference definitions, footnotes, custom
 * containers). The AWS Markdown export is machine-generated and extremely
 * regular — headings, links, anchors, tables, fenced code. Ten regexes cover it
 * and can be explained line by line to a reviewer. If the corpus ever grows a
 * format this mangles, that is the moment to reach for a parser.
 *
 * The judgement calls, in order:
 *
 * - CODE FENCES are dropped entirely, contents included. They are CLI
 *   invocations and JSON policy documents; embedded as prose they add hundreds
 *   of low-signal tokens per page and pull unrelated questions toward whichever
 *   page had the longest example. The prose around them still describes what
 *   they do, which is what a question asks about.
 * - LINK TEXT is kept, link TARGETS are dropped. "see the IAM user guide" is
 *   meaning; "https://docs.aws.amazon.com/IAM/latest/..." is a URL that would
 *   match every other URL in the corpus.
 * - HEADINGS are kept as plain lines. They are the single most informative line
 *   on the page ("Model access") and the chunk containing them retrieves far
 *   better with them than without.
 * - IMAGES are dropped completely; alt text on AWS diagrams is boilerplate.
 */
function markdownToText(markdown: string): string {
  return markdown
    // Fenced code blocks, ``` and ~~~ — whole block, opening fence to closing.
    .replace(/^[ \t]*(```|~~~)[\s\S]*?^[ \t]*\1[ \t]*$/gm, "")
    // An unterminated fence at end of file would otherwise survive the rule above.
    .replace(/^[ \t]*(```|~~~).*$/gm, "")
    // HTML comments, including the AWS "<!-- generated -->" markers.
    .replace(/<!--[\s\S]*?-->/g, "")
    // Anchor tags: <a name="w123"></a> and <a href="...">text</a> — the tag
    // goes, any text between the tags stays.
    .replace(/<a\b[^>]*>/gi, "")
    .replace(/<\/a>/gi, "")
    // Any other stray inline HTML tag (<br/>, <b>, <code>). Content is between
    // tags, never inside them, so removing the tag never removes a word.
    .replace(/<\/?[a-z][a-z0-9-]*\b[^>]*>/gi, "")
    // Images BEFORE links: ![alt](url) shares its shape with [text](url), so a
    // link rule applied first would leave a dangling "!".
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/!\[[^\]]*\]\[[^\]]*\]/g, "")
    // Inline links [text](url) and reference links [text][ref] → text.
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\[[^\]]*\]/g, "$1")
    // Bare autolinks <https://...> → dropped, they carry no prose.
    .replace(/<https?:\/\/[^>]*>/g, "")
    // Reference definitions on their own line: [ref]: https://...
    .replace(/^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$/gm, "")
    // Heading markers: the hashes go, the heading text stays on its own line.
    .replace(/^[ \t]*#{1,6}[ \t]+/gm, "")
    // Setext underlines (=== / ---) and horizontal rules, which would otherwise
    // read as a line of punctuation.
    .replace(/^[ \t]*(?:={3,}|-{3,}|\*{3,}|_{3,})[ \t]*$/gm, "")
    // Blockquote markers and list bullets — the text after them is content.
    .replace(/^[ \t]*>[ \t]?/gm, "")
    .replace(/^[ \t]*[-*+][ \t]+/gm, "")
    // Table pipes and separator rows. Cell text is content; the grid is not.
    .replace(/^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$/gm, "")
    .replace(/[ \t]*\|[ \t]*/g, " ")
    // Inline code: the backticks go, the identifier stays. Parameter names like
    // `modelId` are exactly what a question about the API contains.
    .replace(/`+/g, "")
    // Emphasis markers. Applied last so **bold** inside a link is handled once.
    // Single-character italics are matched for `*` only, never for `_`: the
    // backticks came off one line above, which exposes identifiers like
    // some_parameter_name, and an underscore rule would silently eat the middle
    // of the exact API names a question about the docs contains.
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/\*(?=\S)(.*?)(?<=\S)\*/g, "$1");
}

// ---------------------------------------------------------------------------
// Format dispatch
// ---------------------------------------------------------------------------

/** Extensions this module knows how to read — used in the error message too. */
const SUPPORTED = [".docx", ".pdf", ".xlsx", ".pptx", ".doc", ".xlsb", ".md", ".markdown", ".txt"] as const;

/**
 * Turn one .docx into text that still has its SHAPE.
 *
 * This replaces `mammoth.extractRawText`, and the reason is the most damaging
 * defect found in this pipeline. `extractRawText` walks only paragraphs, so:
 *
 *   - A TABLE loses its geometry entirely. Each cell is emitted as its own
 *     paragraph with nothing marking a cell from a row, so a threshold table
 *     with columns "סוג רכישה | סכום מרבי | גורם מאשר" arrives as
 *     "... 50,000 ועדת הפטור 2,500,000 החשב הכללי ..." — every number present,
 *     no way to tell which number belongs to which authority. Measured on
 *     H.1.4.3.docx: three tables, 49 rows, all flattened.
 *   - CLAUSE NUMBERS never exist. TAKAM numbering is Word auto-numbering held
 *     in numbering.xml, not typed into the text. extractRawText reads only text
 *     nodes, so clauses 1.5.1 … 1.5.9 appear nowhere — while the same document
 *     says "כאמור בסעיף 2.12 להלן". The reference is searchable; its target is
 *     not.
 *
 * That is the mechanism behind the complaint this work started from: the answer
 * IS in the retrieved passage, as a number in a row whose header is gone, or as
 * a clause whose number was never extracted. No amount of prompting recovers
 * information that was destroyed before the model saw it.
 *
 * `convertToHtml` keeps both — mammoth resolves numbering.xml into real
 * <ol>/<li> nesting and emits <table>/<tr>/<th> — so we convert to HTML and
 * walk it back down to text, preserving rows as tab-separated lines and
 * restoring each clause's number from the list nesting.
 */
async function extractDocx(bytes: Buffer): Promise<string> {
  const { value: html } = await mammoth.convertToHtml({ buffer: bytes });
  return htmlToStructuredText(html);
}

/** Minimal entity decoding — mammoth emits only this handful. */
function decodeEntities(t: string): string {
  return t
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#(\d+);/g, (_, d) => String.fromCharCode(Number(d)))
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&");
}

/**
 * Walk mammoth's HTML into plain text that keeps tables and clause numbers.
 *
 * A hand-rolled walk rather than a DOM library because the input is not
 * arbitrary web HTML — it is mammoth's own narrow, well-formed output, and this
 * has to run inside a Lambda where every dependency is bundle weight.
 *
 * The output contract, which the chunker downstream relies on:
 *   - a table row is one LINE, cells separated by TAB
 *   - a list item is one LINE, prefixed with its reconstructed number
 *   - a paragraph is separated from its neighbours by a BLANK line
 */
function htmlToStructuredText(html: string): string {
  const out: string[] = [];
  /** One frame per open list. `ordered` decides whether it contributes a number. */
  const lists: { ordered: boolean; n: number }[] = [];
  let cells: string[] = [];
  let inRow = false;
  let buffer = "";

  /** Text collected since the last tag, cleaned of internal whitespace runs. */
  const take = (): string => {
    const t = decodeEntities(buffer).replace(/\s+/g, " ").trim();
    buffer = "";
    return t;
  };

  /**
   * "1.5.9" — every ordered list currently open, outermost first. An unordered
   * innermost list contributes a bullet instead of a number, so a mixed
   * hierarchy still shows where the item sits.
   */
  const label = (): string => {
    const nums = lists.filter(l => l.ordered).map(l => l.n);
    const innermostUnordered = lists.length > 0 && !lists[lists.length - 1].ordered;
    if (!nums.length) return innermostUnordered ? "•" : "";
    return nums.join(".") + "." + (innermostUnordered ? " •" : "");
  };

  const flushParagraph = () => {
    const t = take();
    if (t) out.push(t);
  };

  // One pass over the tags; everything between them is text.
  const TAG = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = TAG.exec(html)) !== null) {
    buffer += html.slice(last, m.index);
    last = TAG.lastIndex;
    const closing = m[1] === "/";
    const tag = m[2].toLowerCase();

    if (tag === "table") {
      if (!closing) flushParagraph();
      else out.push("");                    // blank line ends the table block
      continue;
    }
    if (tag === "tr") {
      if (!closing) { cells = []; inRow = true; }
      else {
        const cell = take();
        if (cell) cells.push(cell);
        // Tab-separated so the row still reads as a row once it is a flat
        // string, and so a chunk boundary can be placed between rows.
        if (cells.length) out.push(cells.join("\t"));
        inRow = false;
      }
      continue;
    }
    if (tag === "td" || tag === "th") {
      if (closing) { const c = take(); if (c) cells.push(c); }
      continue;
    }
    if (tag === "ol" || tag === "ul") {
      // A nested list opens INSIDE its parent <li>; the parent's own text has
      // to be emitted with the parent's number before the depth changes.
      if (!closing) {
        const t = take();
        if (t) out.push(`${label()} ${t}`.trim());
        lists.push({ ordered: tag === "ol", n: 0 });
      } else {
        lists.pop();
      }
      continue;
    }
    if (tag === "li") {
      if (!closing) {
        if (lists.length) lists[lists.length - 1].n++;
      } else {
        const t = take();
        if (t) out.push(`${label()} ${t}`.trim());
      }
      continue;
    }
    if (tag === "p" || /^h[1-6]$/.test(tag)) {
      // Inside a cell or a list item the enclosing element owns the flush.
      if (closing && !inRow && !lists.length) { flushParagraph(); out.push(""); }
      continue;
    }
    if (tag === "br") { buffer += " "; continue; }
    // Everything else (strong, em, a, span…) is inline: leave the text in place.
  }
  buffer += html.slice(last);
  flushParagraph();

  return out.join("\n");
}

/**
 * Read the text out of an .xlsx workbook.
 *
 * 84 of the ministry's attached files are spreadsheets — price tables, supplier
 * lists, quantity schedules attached to a tender. Skipping them would leave a
 * visible hole exactly where the numbers live.
 *
 * The parse is done by hand against the OOXML rather than by adding a
 * spreadsheet library, for two reasons. JSZip is already present (mammoth
 * depends on it, and it is now a direct dependency so that stays true), and we
 * want one thing out of these files — the words and numbers in reading order.
 * A full workbook model would bring formulas, styling and a much larger Lambda
 * bundle to deliver the same string.
 *
 * Most cell values are not stored in the sheet at all: a `t="s"` cell holds an
 * INDEX into the workbook-wide sharedStrings table, which is why that table is
 * read first. A cell read without it comes back as a bare integer, which embeds
 * as meaningless numerals.
 */
async function extractXlsx(bytes: Buffer): Promise<string> {
  const zip = await JSZip.loadAsync(bytes);

  const decode = (t: string): string =>
    t.replace(/&lt;/g, "<").replace(/&gt;/g, ">")
     .replace(/&quot;/g, '"').replace(/&apos;/g, "'")
     .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(Number(d)))
     .replace(/&amp;/g, "&");

  // The shared string table. Each <si> may be split across several <t> runs
  // when part of the text is styled differently, so the runs are concatenated.
  const shared: string[] = [];
  const sharedFile = zip.file("xl/sharedStrings.xml");
  if (sharedFile) {
    const xml = await sharedFile.async("string");
    for (const si of xml.match(/<si>[\s\S]*?<\/si>/g) ?? []) {
      const runs = [...si.matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map(m => m[1]);
      shared.push(decode(runs.join("")));
    }
  }

  // Numeric sort: as strings, "sheet10" sorts before "sheet2" and the workbook
  // comes out in the wrong order.
  const sheets = Object.keys(zip.files)
    .filter(n => /^xl[/]worksheets[/]sheet\d+[.]xml$/.test(n))
    .sort((a, b) => Number(a.match(/(\d+)/)![1]) - Number(b.match(/(\d+)/)![1]));

  /**
   * A sheet whose XML is larger than this is skipped.
   *
   * Measured 02/09/2026 on AF.a0hKC000000km5EYAQ.xlsx: a 3.4MB workbook whose
   * sheet2.xml decompresses to 47MB — about 14:1 — because Excel had written
   * out a formatted-but-empty grid. It held no more text than its 97KB
   * siblings, and parsing it stalled the ingest with no error at all: the
   * process simply stopped making progress, which is the hardest kind of
   * failure to trace.
   *
   * So this is a guard against a shape of file that really is in this corpus,
   * not a tuning knob. It sits far above any sheet with genuine content here
   * (the largest is 97KB), and the skip is logged rather than silent, because
   * "this workbook is missing a sheet" has to be something a person can learn.
   */
  const MAX_SHEET_CHARS = 4_000_000;

  const out: string[] = [];
  for (const name of sheets) {
    const xml = await zip.file(name)!.async("string");
    if (xml.length > MAX_SHEET_CHARS) {
      console.warn(`   דילוג על ${name}: ${(xml.length / 1e6).toFixed(1)}MB של XML — גיליון מנופח בתאים ריקים`);
      continue;
    }

    // Split on the row terminator instead of matching whole row spans. Split is
    // a single linear pass; `<row[\s\S]*?</row>` allocates a substring per row
    // and, on a sheet whose last row is never closed, scans to the end of the
    // document looking for a terminator that is not there.
    for (const row of xml.split("</row>")) {
      const cells: string[] = [];
      for (const c of row.match(/<c[^>]*(?:[/]>|>[\s\S]*?<[/]c>)/g) ?? []) {
        const type = c.match(/\bt="([^"]+)"/)?.[1];
        if (type === "inlineStr") {
          const runs = [...c.matchAll(/<t[^>]*>([\s\S]*?)<[/]t>/g)].map(m => m[1]);
          if (runs.length) cells.push(decode(runs.join("")));
          continue;
        }
        const v = c.match(/<v>([\s\S]*?)<[/]v>/)?.[1];
        if (v == null) continue;
        cells.push(type === "s" ? (shared[Number(v)] ?? "") : decode(v));
      }
      // Tabs keep a table reading as a table once it is a flat string; empty
      // rows are dropped rather than becoming blank lines.
      const line = cells.filter(x => x.trim()).join("\t");
      if (line) out.push(line);
    }
  }
  return out.join("\n");
}

/**
 * Read the text out of a .pptx deck.
 *
 * One attached file is a slide deck. It is worth the twenty lines because a
 * deck attached to a tender is usually the plain-language summary of it — the
 * part a person would actually ask about.
 *
 * Every piece of text on a slide, whatever shape or table it sits in, is an
 * <a:t> run, so collecting those in document order gives the readable content
 * without modelling slide layout. Slides are sorted numerically: sorting the
 * file names as strings puts slide10 before slide2 and silently scrambles the
 * order of the prose.
 */
async function extractPptx(bytes: Buffer): Promise<string> {
  const zip = await JSZip.loadAsync(bytes);
  const slides = Object.keys(zip.files)
    .filter(n => /^ppt[/]slides[/]slide\d+[.]xml$/.test(n))
    .sort((a, b) => Number(a.match(/(\d+)/)![1]) - Number(b.match(/(\d+)/)![1]));

  const out: string[] = [];
  for (const name of slides) {
    const xml = await zip.file(name)!.async("string");
    const runs = [...xml.matchAll(/<a:t>([^<]*)<[/]a:t>/g)].map(m => m[1]);
    const text = runs.join(" ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").trim();
    if (text) out.push(text);
  }
  return out.join("\n");
}

/**
 * Read the text out of a legacy Word 97-2003 (.doc) file, and out of an .xlsb
 * workbook, by pulling the UTF-16 strings out of the binary.
 *
 * Four forms and annexes in this corpus were published in Word 97 format and
 * one as a binary workbook, and the ministry has exactly one version of each —
 * checked against the API, there is no .docx alternative to fetch instead. So
 * the choice is this or losing them.
 *
 * Both formats store their text as UTF-16LE runs inside a binary container, and
 * decoding the whole file as UTF-16 surfaces those runs interleaved with binary
 * noise. Splitting on characters that cannot occur in prose separates them.
 * Measured 02/09/2026 on all four .doc files, this yields clean Hebrew — "שובר
 * כניסה מס'", "רישום מסירת שוברי תדלוק", "כמות כרטיסים". The obvious
 * alternative, decoding as CP1255, produced garbage on the same files, so the
 * encoding is not a guess.
 *
 * This is deliberately a salvage path and not a parser. It recovers the words;
 * it does not recover reading order across a complex layout, and it cannot know
 * a heading from a cell. For a fill-in form — which is what all five are — the
 * words are the whole content, so that limit costs nothing here. The guard
 * below is what keeps it honest: if too little of what comes out looks like
 * language, we say the file could not be read rather than indexing noise.
 */
function extractUtf16Runs(bytes: Buffer): string {
  const decoded = bytes.toString("utf16le");

  // Split on anything that cannot appear inside a run of prose: control
  // characters, the replacement char that marks an invalid pair, and the
  // private-use area that binary noise decodes into.
  const runs = decoded
    .split(/[^\p{L}\p{N}\p{P}\p{Zs}]+/u)
    .map(r => r.trim())
    .filter(r => r.length >= 4 && /[֐-׿ a-zA-Z]{3}/.test(r));

  // Consecutive duplicates are a signature of binary padding decoding to the
  // same glyph, not of a document that repeats itself.
  const out: string[] = [];
  for (const r of runs) if (r !== out[out.length - 1]) out.push(r);

  const text = out.join("\n");

  /**
   * The honesty guard. A salvage that returns mostly punctuation and stray
   * letters has not read the document, and the caller must be able to tell the
   * difference between "this file is empty" and "this approach did not work
   * here". Below this bar we return nothing, and the ingest reports the
   * document as failed instead of indexing rubble.
   */
  const letters = (text.match(/[֐-׿ a-zA-Z]/g) ?? []).length;
  if (!text || letters / text.length < 0.5) return "";
  return text;
}

/**
 * Read the text layer out of a PDF.
 *
 * 51 attached files are PDFs. What comes back is only as good as the file: a
 * born-digital document extracts cleanly, while one produced from a design tool
 * can come back with its letters spaced apart, and a scan has no text layer at
 * all and yields nothing. That is a property of the source, not a bug here —
 * but it is why the caller treats an empty extraction as a failed document
 * rather than as an empty one.
 */
async function extractPdf(bytes: Buffer): Promise<string> {
  const parser = new PDFParse({ data: new Uint8Array(bytes) });
  try {
    const { text } = await parser.getText();
    return text;
  } finally {
    // pdf-parse holds a worker open; leaking one per document would exhaust the
    // process long before 51 files are done.
    await parser.destroy();
  }
}

/**
 * Read one source file into plain text.
 *
 * Returns an object rather than a bare string so a future format can report
 * extra findings (page count, mammoth's conversion warnings) without every
 * caller changing shape.
 *
 * .docx goes through `mammoth.extractRawText`, which is the path already
 * verified against all 18 Hebrew chapter-7 files. Mammoth reads the OOXML
 * directly — no Word, no LibreOffice, no shelling out — which is what lets the
 * same code run unattended inside a Lambda.
 */
export async function extractText(
  fileName: string,
  bytes: Buffer,
): Promise<{ text: string }> {
  const ext = extensionOf(fileName);

  switch (ext) {
    case ".docx":
      return { text: normalise(await extractDocx(bytes)) };

    case ".pdf":
      return { text: normalise(await extractPdf(bytes)) };

    case ".xlsx":
      return { text: normalise(await extractXlsx(bytes)) };

    case ".pptx":
      return { text: normalise(await extractPptx(bytes)) };

    case ".doc":
    case ".xlsb": {
      const salvaged = normalise(extractUtf16Runs(bytes));
      // Empty here does not mean "an empty document" — it means the salvage
      // found nothing that reads as language. Saying so out loud is the point:
      // a silently empty document would be indexed as a real one with no text,
      // and would then be missing from answers with nothing to point at.
      if (!salvaged) {
        throw new Error(
          `לא ניתן לחלץ טקסט מ-${baseName(fileName)}: פורמט ישן (${ext}) שאין לו גרסה מודרנית אצל המשרד, ` +
          `וחילוץ המחרוזות לא החזיר תוכן קריא.`,
        );
      }
      return { text: salvaged };
    }

    case ".md":
    case ".markdown":
      return { text: normalise(markdownToText(bytes.toString("utf-8"))) };

    case ".txt":
      return { text: normalise(bytes.toString("utf-8")) };

    default:
      // A loud failure on purpose. The alternative — returning the raw bytes as
      // text — would put PDF or ZIP binary into the index, where it embeds
      // successfully, retrieves occasionally, and is never traced back here.
      throw new Error(
        `סיומת קובץ לא נתמכת: "${ext || "ללא סיומת"}" (קובץ: ${baseName(fileName)}). ` +
        `נתמכים: ${SUPPORTED.join(", ")}`,
      );
  }
}

// ---------------------------------------------------------------------------
// Citation handles
// ---------------------------------------------------------------------------

/**
 * The short code shown under an answer, derived from the file name.
 *
 * The rule is chosen by the collection's `citation` field in config.ts, NOT by
 * its id. That indirection is the whole point: a new AWS service collection
 * (`aws-dynamodb`, `aws-sqs`, ...) declares `citation: "page"` in the config
 * and this function already handles it. Matching on `collection === "takam"`
 * would mean editing this file every time a collection is added — the exact
 * coupling AD-4 exists to avoid.
 *
 *   "instruction" — TAKAM. "docs/takam/H.7.6.1.docx" → "7.6.1"
 *                   The "H." prefix is the ministry's own file-naming scheme
 *                   (H for הוראה); the citation the user knows is the number.
 *   "page"        — AWS docs. "docs/aws-bedrock/model-access.md" → "model-access"
 *                   The slug IS the citation: it is the last path segment of
 *                   the public docs URL, so a reader can find the page from it.
 */
export function deriveCode(fileName: string, collection: string): string {
  const stem = stripExtension(fileName);

  switch (collectionDef(collection).citation) {
    case "instruction":
      return stem.replace(/^H\./i, "");
    case "page":
    default:
      return stem;
  }
}

/**
 * The human-readable title shown in the sources box.
 *
 * Two very different situations, again keyed off `citation`:
 *
 * - TAKAM has an authoritative title from the catalogue API, carried in
 *   corpus/manifest.json ("הגדרות בנושא התקשרויות ורכישות"). Nothing in the
 *   .docx beats it, so the caller passes it in as `catalogTitle` and it wins.
 * - AWS Markdown has no manifest, but every page opens with an H1 that is the
 *   real page title ("Model access"), which is far more readable than the slug
 *   "model-access". So we read it out of the source.
 *
 * IMPORTANT: pass the RAW markdown here, before extractText() has run. The
 * extraction deliberately turns headings into plain lines, so once it has run
 * the H1 is no longer distinguishable from any other line.
 *
 * Fallback order is always: catalogue title → H1 → humanised slug → code. The
 * last one can never be empty, so the sources box never renders a blank row.
 */
export function titleFor(args: {
  fileName: string;
  collection: string;
  code: string;
  /** Raw file contents, pre-extraction. Only used for Markdown. */
  rawText?: string;
  /** Authoritative title from the corpus manifest, when one exists. */
  catalogTitle?: string;
}): string {
  const { fileName, collection, code, rawText, catalogTitle } = args;

  if (catalogTitle?.trim()) return catalogTitle.trim();

  const ext = extensionOf(fileName);
  if ((ext === ".md" || ext === ".markdown") && rawText) {
    // First level-1 heading only. Deeper headings are section names, not the
    // page title, and would mislabel the source.
    const h1 = rawText.match(/^[ \t]*#[ \t]+(.+?)[ \t]*#*[ \t]*$/m);
    if (h1) {
      const cleaned = markdownToText(h1[1]).trim();
      if (cleaned) return cleaned;
    }
  }

  // Humanised slug: "model-access" → "model access". Not pretty, but it reads
  // as words rather than as an identifier.
  const humanised = code.replace(/[-_]+/g, " ").trim();
  return humanised || code;
}
