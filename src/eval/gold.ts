/**
 * gold.ts — the shapes and the string normaliser behind the accuracy ruler.
 *
 * WHY THIS FILE EXISTS
 * Until now the project measured ONE thing well: did the search put the right
 * instruction first (74%). But the model reads five instructions, so the number
 * that actually matters — did the ANSWER carry the right fact and cite the right
 * document — has never been measured at all. Every improvement in stages 1-5 of
 * the accuracy plan is a guess until that number exists.
 *
 * A gold record is therefore not "a question with a known document". It is a
 * question with a known FACT: the exact string that must appear in a correct
 * answer. That single field is what turns a retrieval benchmark into an answer
 * benchmark, and it is also what makes the record verifiable by a human in
 * fifteen seconds against the ministry's own website.
 *
 * The normaliser lives here rather than in the judge because two very different
 * callers must agree on it byte for byte: the GENERATOR uses it to refuse a
 * record whose fact cannot be found in the source instruction, and the JUDGE
 * uses it to decide whether that same fact appeared in the answer. If the two
 * ever drift, records would be accepted that the judge can never credit — the
 * measurement would fail silently, in the direction of looking worse than it is.
 */

import { readFileSync, existsSync } from "node:fs";
import { REFUSAL_PREFIX as CORE_REFUSAL_PREFIX, REFUSAL_LEAD as CORE_REFUSAL_LEAD, isRefusalText } from "../core/answer.js";

/** The five question shapes the plan asks the gold set to cover. */
export const GOLD_KINDS = ["מספרי", "כן/לא", "מי-מאשר", "איפה-כתוב", "תהליך"] as const;
export type GoldKind = (typeof GOLD_KINDS)[number];

/** One measurable question. */
export interface GoldRecord {
  /** Stable id, e.g. "g-042". Used to join results across runs. */
  id: string;
  /** The question, in citizen wording. Never contains the instruction number. */
  q: string;
  /** The instruction that answers it, e.g. "7.6.1". */
  expect: string;
  chapter: number;
  /** Clause inside the instruction, e.g. "3.2". Helps a human verify fast. */
  clause: string;
  /**
   * Whether that clause number was actually found in the instruction text.
   *
   * Recorded rather than enforced. Rejecting on it threw away 21% of otherwise
   * valid records in the trial run — records whose `fact` was already proven to
   * be in the document. The clause is a pointer for the human reviewer, so a
   * wrong pointer degrades convenience, not correctness; the review sheet shows
   * an unverified clause in grey so the reviewer knows to search instead.
   */
  clauseVerified?: boolean;
  /**
   * The string a correct answer must contain — a threshold, a percentage, an
   * approving authority, a form name. Short and literal, quoted from the
   * instruction, because a long paraphrase cannot be checked mechanically and a
   * judge that has to interpret is a judge that drifts.
   */
  fact: string;
  kind: GoldKind;
  /** Instruction title, shown in the human review sheet. */
  title: string;
  /** Where the record came from: model-generated, or one of the original 76. */
  source: "generated" | "covered-76";
  /** Deep link to the ministry's page, so a human can check in one click. */
  sourceUrl?: string;
  /**
   * How many times `fact` occurs in its own source instruction.
   *
   * Discriminative power, measured once at build time. A value appearing once
   * is a specific answer; one appearing 58 times is not.
   */
  factOccurrences?: number;
  /**
   * True when a literal match on `fact` would be worthless evidence.
   *
   * Measured across the set: 23 of 272 records (8.5%) carry a fact that occurs
   * eight or more times in its own instruction, and the worst are "לא" (58
   * occurrences), "כן" (23) and "3" (19). Every Hebrew answer contains the word
   * "לא" somewhere, so a literal match would have credited those records as
   * correct no matter what the system replied — inflating answer accuracy by up
   * to 8.5 points, in the flattering direction.
   *
   * The records are NOT deleted, because a yes/no question legitimately has "כן"
   * as its answer and dropping them would bias the set away from a whole `kind`.
   * Instead the judge skips its literal shortcut for them and asks the model
   * "was this conveyed?", which is a real question about "כן" and a meaningless
   * one for a substring search.
   */
  weakFact?: boolean;
  /**
   * True when the fact could not be located verbatim in the instruction text.
   *
   * Only ever set on the imported 76: those questions predate this schema and
   * their wording is frozen (they are the comparison baseline for every
   * measurement since 06/09), so a failed fact lookup must not delete them.
   * They still count for recall and ranking; they are excluded from the answer
   * accuracy denominator, and the exclusion is reported rather than hidden.
   */
  needsReview?: boolean;
}

/** One adjacent-domain question the system must refuse. */
export interface ControlRecord {
  id: string;
  q: string;
  /** Which neighbouring domain it belongs to — national insurance, tax, ... */
  domain?: string;
  source: "uncovered-50" | "generated";
  /**
   * How the generated control was proven to be outside the corpus: the model
   * read the top passages the search returned and said they do not answer it.
   * Absent for the original 50, which were hand-written.
   */
  verified?: boolean;
}

/**
 * Normalise a Hebrew string for literal comparison.
 *
 * Everything here is a real difference observed between how a regulation writes
 * a value and how an answer repeats it. None of it is cosmetic:
 *   - Hebrew geresh/gershayim (׳ ״) against ASCII quotes: "אית״ן" vs "אית'ן".
 *   - Curly quotes, which arrive from Word documents and never from a model.
 *   - The Hebrew maqaf (־) against a plain hyphen, in ranges like "3-5".
 *   - Niqqud, which appears in a handful of older instructions.
 *   - Percentages: an instruction writes "7.5%" and an answer often "7.5 אחוז"
 *     — those stay different on purpose (the judge falls through to the model
 *     for that case); what is folded is only the SPACE in "7.5 %".
 *   - Thousands separators: "10,000" against "10000".
 * Deliberately NOT folded: final letter forms (ם/מ), which change words, and
 * digits, which are the whole point of a numeric fact.
 */
export function normalise(s: string): string {
  return s
    .normalize("NFKC")
    /**
     * Niqqud and cantillation, in TWO ranges rather than one.
     *
     * The obvious single range U+0591-U+05C7 is wrong, and wrong in a way that
     * shows up as a wrong ANSWER rather than as an error: the Hebrew maqaf
     * (U+05BE, the hyphen in "3־5") sits inside it. Stripping it turned the
     * range "3־5" into the number "35", so a gold fact of "35" would have
     * matched a passage that says three-to-five. Measured, not theorised — the
     * first run of the normaliser test failed on exactly this case.
     */
    .replace(/[֑-ֽ]/g, "")                     // niqqud/cantillation, up to U+05BD
    .replace(/[ֿ-ׇ]/g, "")                     // and from U+05BF on — U+05BE (maqaf) is punctuation, kept
    .replace(/[׳‘’ʼ]/g, "'")         // geresh family -> ASCII apostrophe
    .replace(/[״“”„]/g, '"')         // gershayim family -> ASCII quote
    .replace(/[־‐-―]/g, "-")                 // maqaf + dashes -> hyphen
    .replace(/(\d)\s*,\s*(\d{3})\b/g, "$1$2") // 10,000 -> 10000
    .replace(/\s*%/g, "%")                    // "7.5 %" -> "7.5%"
    .replace(/\s+/g, " ")
    .trim();
}

/** True when `needle` appears inside `hay`, both normalised. */
export function containsNormalised(hay: string, needle: string): boolean {
  const n = normalise(needle);
  return n.length > 0 && normalise(hay).includes(n);
}

/**
 * Refusal detection is NOT defined here — it is imported from the core.
 *
 * It briefly was defined here, and that was the wrong call: the same test also
 * decides, in production, whether an answer gets a sources box. Two copies of a
 * rule that must agree is how a measurement and the thing it measures drift
 * apart. The core owns it; the evaluator asks.
 */
export const REFUSAL_PREFIX = CORE_REFUSAL_PREFIX;
export const isRefusal = isRefusalText;
/** The markdown a model puts in front of a refusal. Re-exported so the judge can strip it. */
export const REFUSAL_LEAD = CORE_REFUSAL_LEAD;

/** Loose: the phrase appears anywhere. This is what the 07/09 baseline counted. */
export const mentionsRefusal = (text: string): boolean => text.includes(REFUSAL_PREFIX);

/** A TAKAM instruction code: three dot-separated numbers, e.g. "7.6.1". */
export const CODE_RE = /\b\d{1,2}\.\d{1,2}\.\d{1,3}\b/;

export function loadGold(path: string | undefined = "eval/gold/gold.json"): GoldRecord[] {
  path = path ?? "eval/gold/gold.json";
  if (!existsSync(path)) throw new Error(`סט הזהב לא נמצא: ${path} — יש להריץ npm run eval:gen-gold`);
  return JSON.parse(readFileSync(path, "utf8")) as GoldRecord[];
}

export function loadControls(path = "eval/controls.json"): ControlRecord[] {
  if (!existsSync(path)) throw new Error(`קובץ הבקרות לא נמצא: ${path} — יש להריץ npm run eval:gen-controls`);
  return JSON.parse(readFileSync(path, "utf8")) as ControlRecord[];
}
