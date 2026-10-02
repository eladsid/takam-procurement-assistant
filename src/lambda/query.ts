/**
 * lambda/query.ts — the question handler, behind a Function URL (FR-2.1).
 *
 * A thin wrapper around the same core the local server calls (AD-1): parse the
 * HTTP envelope, validate, delegate, translate errors into status codes.
 *
 * Its IAM role can read `index/` and invoke models — and nothing else (AD-5).
 * It cannot write to the bucket at all, so even a total compromise of this
 * function cannot corrupt the corpus. That is a small, demonstrable claim, and
 * demonstrable beats impressive.
 */

import { ask } from "../core/answer.js";
import { COLLECTIONS, DEFAULT_COLLECTION } from "../core/config.js";
import { collectionAvailable } from "../core/store.js";

/** The slice of the Function URL event shape we rely on. */
interface FunctionUrlEvent {
  requestContext?: { http?: { method?: string; path?: string } };
  rawPath?: string;
  body?: string;
  isBase64Encoded?: boolean;
}

interface LambdaResponse {
  statusCode: number;
  headers: Record<string, string>;
  body: string;
}

/**
 * The page is served from a different origin than the Function URL, so the
 * browser will not read the response without these. Kept permissive because
 * the API is public-read by design for the POC — there is nothing behind it to
 * protect, and every document it quotes is already published by the ministry.
 * Production would put API Gateway plus organisational SSO in front (see
 * "what changes in production" in the architecture document).
 */
const CORS: Record<string, string> = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "POST, GET, OPTIONS",
  "access-control-allow-headers": "content-type",
  "access-control-max-age": "86400",
};

const json = (statusCode: number, body: unknown): LambdaResponse => ({
  statusCode,
  headers: { "content-type": "application/json; charset=utf-8", ...CORS },
  body: JSON.stringify(body),
});

export const handler = async (event: FunctionUrlEvent): Promise<LambdaResponse> => {
  const method = event.requestContext?.http?.method ?? "POST";
  const path = event.requestContext?.http?.path ?? event.rawPath ?? "/";

  // The browser's CORS preflight. Answering it costs one warm invocation.
  if (method === "OPTIONS") return { statusCode: 204, headers: CORS, body: "" };

  // Which corpora actually have an index built. The UI asks before rendering
  // its picker, so a collection that would answer nothing is never offered.
  if (method === "GET" && path.endsWith("/collections")) {
    const available = [];
    for (const c of COLLECTIONS) {
      if (await collectionAvailable(c)) available.push(c);
    }
    /**
     * The advertised default must be a collection the picker actually lists.
     *
     * This returned DEFAULT_COLLECTION unconditionally. `takam-all` is virtual,
     * and collectionAvailable hides it until at least two member indexes exist
     * — so in a fresh environment the response named a default that was missing
     * from its own list. The UI then showed the first option while still
     * POSTing the advertised id, and the server threw "no index for takam-all"
     * on the very first question, behind a dropdown that looked correct.
     */
    const fallback = available.some(c => c.id === DEFAULT_COLLECTION)
      ? DEFAULT_COLLECTION
      : available[0]?.id ?? COLLECTIONS[0].id;

    return json(200, {
      collections: available.length ? available : [COLLECTIONS[0]],
      default: fallback,
    });
  }

  if (method !== "POST") return json(405, { error: "שיטה לא נתמכת" });

  try {
    const raw = event.isBase64Encoded && event.body
      ? Buffer.from(event.body, "base64").toString("utf-8")
      : event.body ?? "{}";
    const parsed = JSON.parse(raw);

    const question = String(parsed.question ?? "").trim();
    const collection = String(parsed.collection ?? DEFAULT_COLLECTION);

    if (question.length < 3) return json(400, { error: "השאלה קצרה מדי" });
    if (question.length > 500) return json(400, { error: "השאלה ארוכה מדי (עד 500 תווים)" });

    const answer = await ask(question, collection);

    // Structured so CloudWatch Logs Insights can chart cost and latency per
    // question. The question text itself is logged: this corpus is public and
    // there is no personal data, and seeing the real questions is how the
    // retrieval quality gets tuned. A corpus with citizen data would not.
    console.log(
      JSON.stringify({
        level: "info",
        event: "answered",
        collection,
        question,
        cited: answer.sources.map(s => s.code),
        // The structural flag, not `sources.length === 0`: that proxy had already
        // drifted from answer.ts, and a refusal that keeps its quoted citations
        // (REFUSAL_KEEPS_QUOTED_SOURCES) would have been charted as answered.
        refused: answer.refused,
        route: answer.route,
        costUSD: Number(answer.usage.costUSD.toFixed(6)),
      }),
    );

    return json(200, answer);
  } catch (err: any) {
    const throttled = err?.name === "ThrottlingException" || err?.$metadata?.httpStatusCode === 429;
    console.error(
      JSON.stringify({
        level: "error",
        event: "query_failed",
        error: err?.name ?? "Error",
        message: err?.message ?? String(err),
      }),
    );
    // The client gets a name and a next step, never a stack trace.
    return json(throttled ? 429 : 500, {
      error: throttled
        ? "המודל עמוס כרגע (מכסה). נסה שוב בעוד רגע."
        : `שגיאה בשרת: ${err?.name ?? "לא ידועה"}`,
    });
  }
};
