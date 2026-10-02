/**
 * clarify.ts — the layer that asks the user WHICH TOPIC they meant.
 *
 * Why this exists, in measured terms. On 76 questions written the way a person
 * actually types them, the top document answers the question 47% of the time,
 * while one of the top THREE documents answers it 66% of the time. Nineteen
 * points sit in the gap between "the system's first guess" and "the system's
 * shortlist" — and the only thing that can close it reliably is the person who
 * asked, because they know what they meant and the retriever does not.
 *
 * The same data killed the obvious alternative. A confident score is not a
 * correct one: even above 0.55 the first document answers only 55% of the time.
 * So clarification is not a fallback for the hesitant case; it belongs on most
 * questions that touch more than one topic.
 *
 * Two rules shape everything here:
 *
 *   1. TOPICS, NOT NUMBERS. A person does not know that "1.4.5" is the one they
 *      want; they know they meant advance payments. Instruction codes are an
 *      internal handle and appear only in the citation of a final answer.
 *
 *   2. CLARIFYING IS NOT ANSWERING. Offering topics must never become a way to
 *      answer a question the corpus does not cover. Every clarification carries
 *      an explicit "none of these", and the refusal path stays reachable.
 */

import type { Hit } from "./types.js";
import {
  CLARIFY_MIN_SCORE, CLARIFY_MAX_TOPICS, CLARIFY_ALWAYS, minScoreFor,
} from "./config.js";

/** One choice offered to the user. `codes` never reaches the screen. */
export interface Topic {
  /** What the user reads: the document's own official title. */
  label: string;
  /** The documents this choice resolves to. Internal. */
  codes: string[];
  /** Best semantic score among them — for ordering and for the trace. */
  score: number;
}

export interface Clarification {
  question: string;
  topics: Topic[];
  /** The literal option that must always be offered. */
  noneLabel: string;
}

export const NONE_OF_THESE = "אף אחד מאלה";

/**
 * Normalise a title for duplicate detection.
 *
 * Chapter 14 alone holds "ימי חופשה של עובד איתן", "ימי אבל" and "ימי אבל
 * איתן". Offering all three as separate choices asks the reader to distinguish
 * between things they cannot distinguish from the outside, which is friction
 * rather than help.
 */
const norm = (t: string): string =>
  t.replace(/["'״׳()]/g, "").replace(/\s+/g, " ").trim();

/** Do two titles describe the same thing closely enough to merge? */
function sameTopic(a: string, b: string): boolean {
  const [x, y] = [norm(a), norm(b)];
  if (x === y) return true;
  if (x.includes(y) || y.includes(x)) return true;
  // Word overlap: 3+ shared content words on titles this short means one topic.
  const w = (s: string) => new Set(s.split(" ").filter(t => t.length >= 3));
  const [wx, wy] = [w(x), w(y)];
  let shared = 0;
  for (const t of wx) if (wy.has(t)) shared++;
  return shared >= 3 && shared >= Math.min(wx.size, wy.size) - 1;
}

/**
 * Turn retrieved passages into the shortlist of topics to offer.
 *
 * Grouping is by DOCUMENT, not by passage: three chunks of one instruction are
 * one topic, and a reader offered "instruction 7.6.1, passage 3" learns
 * nothing. Lexically-assisted passages are excluded — they were pulled in by a
 * phrase matching a table row, which is a reason to show a table to the model,
 * not a reason to tell a person that this is what they probably meant.
 */
export function buildTopics(hits: Hit[], max = CLARIFY_MAX_TOPICS): Topic[] {
  const byDoc = new Map<string, { title: string; score: number }>();
  for (const h of hits) {
    if (h.lexical) continue;
    const prev = byDoc.get(h.code);
    if (!prev || h.score > prev.score) {
      byDoc.set(h.code, { title: h.title || h.code, score: h.score });
    }
  }

  const topics: Topic[] = [];
  for (const [code, { title, score }] of [...byDoc.entries()].sort((a, b) => b[1].score - a[1].score)) {
    const existing = topics.find(t => sameTopic(t.label, title));
    if (existing) { existing.codes.push(code); continue; }
    if (topics.length >= max) continue;
    topics.push({ label: title, codes: [code], score });
  }
  return topics;
}

/**
 * Should this question be answered directly, clarified, or refused?
 *
 * "clarify" is deliberately the widest band. The measurement behind that: the
 * 0.40–0.55 band held 41 of 76 real questions — more than the confident and the
 * hopeless bands combined — and inside it the right document was among the top
 * three for 71% of them. That is the population this layer exists to serve.
 */
export type Decision = "answer" | "clarify" | "refuse";

export function decide(hits: Hit[], collection: string): Decision {
  const semantic = hits.filter(h => !h.lexical);
  const best = semantic[0]?.score ?? 0;

  // Below the refusal threshold minus the clarify window, nothing is worth
  // offering: showing three unrelated topics is a worse answer than "I did not
  // find it", because it implies the corpus is relevant when it is not.
  if (best < CLARIFY_MIN_SCORE) return "refuse";

  // More than one distinct topic in play means the question is ambiguous
  // regardless of how high the top score is — the 55%-correct-when-confident
  // measurement is what makes this not merely cautious.
  const topics = buildTopics(hits);
  if (topics.length <= 1) return "answer";

  if (best >= minScoreFor(collection) && !CLARIFY_ALWAYS) return "answer";
  return "clarify";
}

/** Render the question the user sees. No model call, so nothing to hallucinate. */
export function buildClarification(question: string, hits: Hit[]): Clarification {
  return { question, topics: buildTopics(hits), noneLabel: NONE_OF_THESE };
}
