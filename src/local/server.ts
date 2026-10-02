/**
 * server.ts — the local run mode (RUN_MODE=local), the demo's safety net.
 *
 * This is not a mock. It imports exactly the same core modules the Lambda
 * imports; the only difference is where documents and indexes are read from
 * (AD-6). If the network dies, or the Function URL misbehaves during a demo,
 * the same product still answers the same questions.
 *
 * The browser never holds an AWS credential and never talks to AWS: it talks to
 * this process, and this process talks to the cloud. That is the same shape as
 * the Lambda deployment, which is why the front end needs no changes to switch.
 *
 * Run:  npm run serve
 */

import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { ask } from "../core/answer.js";
import { COLLECTIONS, DEFAULT_COLLECTION, describeConfig, indexKeyFor, collectionDef } from "../core/config.js";
import { collectionAvailable, loadIndex, resolveMembers } from "../core/store.js";
import { warmUnion } from "../core/retrieve.js";

const POC_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const PUBLIC_DIR = join(POC_ROOT, "public");
const PORT = Number(process.env.PORT ?? 5173);

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function sendJson(res: import("node:http").ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

/** Read a JSON request body with a hard size cap — an unbounded read is a trivial DoS. */
async function readBody(req: import("node:http").IncomingMessage, maxBytes = 64 * 1024): Promise<string> {
  const parts: Buffer[] = [];
  let size = 0;
  for await (const part of req) {
    size += (part as Buffer).length;
    if (size > maxBytes) throw new Error("גוף הבקשה גדול מדי");
    parts.push(part as Buffer);
  }
  return Buffer.concat(parts).toString("utf-8");
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);

  // --- static files -------------------------------------------------------
  if (req.method === "GET" && !url.pathname.startsWith("/api/")) {
    const rel = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    // Resolve first, then confirm the result is still inside PUBLIC_DIR, so a
    // crafted path like ../../.env cannot escape the served directory.
    const file = resolve(PUBLIC_DIR, rel);
    if (!file.startsWith(PUBLIC_DIR) || !existsSync(file)) {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      return res.end("לא נמצא");
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    return res.end(readFileSync(file));
  }

  // --- which corpora are actually available -------------------------------
  // The UI asks before rendering its picker. Only collections with a built
  // index are offered: showing a corpus that answers nothing is worse than
  // not showing it, especially in a live demo.
  if (req.method === "GET" && url.pathname === "/api/collections") {
    const available = [];
    for (const c of COLLECTIONS) {
      if (await collectionAvailable(c)) available.push(c);
    }
    /**
     * The advertised default must be a collection the picker actually lists.
     *
     * This returned DEFAULT_COLLECTION unconditionally. `takam-all` is virtual,
     * and collectionAvailable hides it until at least two member indexes exist,
     * so in a fresh environment the response named a default that was missing
     * from its own list. The UI then showed the first option while still
     * POSTing the advertised id, and the server threw "no index for takam-all"
     * on the very first question, behind a dropdown that looked correct.
     */
    const fallback = available.some(c => c.id === DEFAULT_COLLECTION)
      ? DEFAULT_COLLECTION
      : available[0]?.id ?? COLLECTIONS[0].id;

    return sendJson(res, 200, {
      collections: available.length ? available : [COLLECTIONS[0]],
      default: fallback,
    });
  }

  // --- the question -------------------------------------------------------
  if (req.method === "POST" && url.pathname === "/api/ask") {
    try {
      const parsed = JSON.parse(await readBody(req));
      const question = String(parsed.question ?? "").trim();
      const collection = String(parsed.collection ?? DEFAULT_COLLECTION);

      if (question.length < 3) return sendJson(res, 400, { error: "השאלה קצרה מדי" });
      if (question.length > 500) return sendJson(res, 400, { error: "השאלה ארוכה מדי (עד 500 תווים)" });

      /**
       * `topicCodes` arrives on the SECOND request of a clarification: the user
       * read the topics, picked one, and the client sends back the codes that
       * topic stood for. The codes never appear on screen — they are the
       * internal handle behind a label like "תשלום מקדמות".
       */
      const topicCodes = Array.isArray(parsed.topicCodes)
        ? parsed.topicCodes.map(String).slice(0, 8)
        : undefined;
      const topicLabel = parsed.topicLabel ? String(parsed.topicLabel).slice(0, 200) : undefined;

      return sendJson(res, 200, await ask(question, collection, undefined, { topicCodes, topicLabel }));
    } catch (err: any) {
      // Throttling is a capacity signal the user can act on ("try again in a
      // moment"); everything else is ours to fix, so the client gets the error
      // name only and the stack stays in the server log.
      const throttled = err?.name === "ThrottlingException" || err?.$metadata?.httpStatusCode === 429;
      console.error("שגיאה:", err?.name ?? err, err?.message ?? "");
      return sendJson(res, throttled ? 429 : 500, {
        error: throttled
          ? "המודל עמוס כרגע (מכסה). נסה שוב בעוד רגע."
          : `שגיאה בשרת: ${err?.name ?? "לא ידועה"}`,
      });
    }
  }

  /**
   * The same question, streamed.
   *
   * A separate ROUTE but not separate LOGIC: it calls the same `ask` with a
   * callback. `/api/ask` above is left untouched, so the blocking path stays
   * available as a fallback and the Lambda, which cannot stream over a plain
   * Function URL response, keeps working unchanged.
   *
   * Server-Sent Events rather than WebSocket: the traffic is one-directional
   * and short-lived, SSE is plain HTTP that needs no upgrade handshake, and it
   * survives the corporate proxies a government network is likely to sit
   * behind. It is POST, not GET, so the browser's EventSource cannot be used —
   * the front end reads the response body as a stream instead.
   */
  if (req.method === "POST" && url.pathname === "/api/ask-stream") {
    let opened = false;
    try {
      const parsed = JSON.parse(await readBody(req));
      const question = String(parsed.question ?? "").trim();
      const collection = String(parsed.collection ?? DEFAULT_COLLECTION);

      if (question.length < 3) return sendJson(res, 400, { error: "השאלה קצרה מדי" });
      if (question.length > 500) return sendJson(res, 400, { error: "השאלה ארוכה מדי (עד 500 תווים)" });

      res.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        // Named for nginx, harmless everywhere else. Without it a buffering
        // proxy holds the whole stream and delivers it at the end — which
        // looks exactly like the blocking version this route exists to
        // replace, and would be invisible in local testing.
        "x-accel-buffering": "no",
      });
      opened = true;

      const send = (event: string, data: unknown) =>
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

      // The streamed route accepts a topic choice too, so the second half of a
      // clarification streams exactly like the first half. A clarification
      // itself streams no tokens — ask() returns before the model is called —
      // and simply arrives as the `done` frame the client already handles.
      const topicCodes = Array.isArray(parsed.topicCodes)
        ? parsed.topicCodes.map(String).slice(0, 8)
        : undefined;
      const topicLabel = parsed.topicLabel ? String(parsed.topicLabel).slice(0, 200) : undefined;

      const answer = await ask(question, collection, delta => send("token", delta), { topicCodes, topicLabel });

      // The full Answer object, exactly as /api/ask returns it. The client
      // replaces the text it accumulated with `answer.text` — one authoritative
      // string, so a dropped or duplicated fragment cannot leave the sources
      // box describing a different answer from the one on screen.
      send("done", answer);
      return res.end();
    } catch (err: any) {
      const throttled = err?.name === "ThrottlingException" || err?.$metadata?.httpStatusCode === 429;
      const message = throttled
        ? "המודל עמוס כרגע (מכסה). נסה שוב בעוד רגע."
        : `שגיאה בשרת: ${err?.name ?? "לא ידועה"}`;
      console.error("שגיאה:", err?.name ?? err, err?.message ?? "");

      // Once the 200 and the SSE headers are out, the status code can no longer
      // carry the failure — the only channel left is an event on the open
      // stream. Sending JSON here would produce a body the client parses as
      // events and silently ignores.
      if (opened) {
        res.write(`event: error\ndata: ${JSON.stringify({ error: message })}\n\n`);
        return res.end();
      }
      return sendJson(res, throttled ? 429 : 500, { error: message });
    }
  }

  res.writeHead(405, { "content-type": "text/plain; charset=utf-8" });
  res.end("שיטה לא נתמכת");
});

/**
 * Load the default collection's indexes now, instead of on the first question.
 *
 * Measured 03/09/2026: the sixteen TAKAM indexes take 1.6 seconds to read and
 * parse, and `loadIndex` caches per collection — so that cost was paid exactly
 * once, by whoever asked first. In a demo that person is the audience. Warming
 * here moves the 1.6 seconds into the seconds after `npm run serve`, where
 * nobody is watching.
 *
 * Deliberately NOT awaited before `listen`: a warm-up that delays the port
 * would turn a fast start into a slow one, and a warm-up that throws would stop
 * a server that is otherwise perfectly able to answer. A failure here is logged
 * and ignored, because the first question would load the index anyway.
 */
async function warmIndexes(): Promise<void> {
  const started = Date.now();
  const def = collectionDef(DEFAULT_COLLECTION);
  const members = await resolveMembers(def);
  const targets = members.length ? members : [def.id];
  await Promise.all(targets.map(m => loadIndex(m)));
  console.log(`  אינדקסים נטענו מראש: ${targets.length} (${Date.now() - started}ms) — השאלה הראשונה כבר לא משלמת על זה`);
  // The whole-corpus BM25 engine (retrieve.ts, unionOf) indexes all 28K chunks
  // once. Built here it costs the seconds after startup; built lazily it would
  // cost the first person who asks, and in a demo that is the audience.
  if (targets.length > 1) {
    const t1 = Date.now();
    await warmUnion(targets);
    console.log(`  מנוע BM25 מאוחד נבנה מראש: ${targets.length} אוספים (${Date.now() - t1}ms)`);
  }
}

server.listen(PORT, () => {
  console.log(`\nעוזר הנהלים — מצב מקומי`);
  console.log(`  ${describeConfig()}`);
  console.log(`  אינדקס ברירת מחדל: ${indexKeyFor(DEFAULT_COLLECTION)}`);
  console.log(`\n  http://localhost:${PORT}\n`);
  void warmIndexes().catch(err =>
    console.error(`  (חימום האינדקסים נכשל, לא קריטי: ${err?.name ?? err})`),
  );
});
