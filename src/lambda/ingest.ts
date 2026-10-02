/**
 * lambda/ingest.ts — the live-ingest handler (FR-1.1).
 *
 * Triggered by an S3 ObjectCreated event: a document lands in the bucket, this
 * function extracts it, chunks it, embeds it, and merges it into that
 * collection's index. Nothing schedules it and nothing polls — the storage
 * layer itself is the trigger, which is the difference between "a script I run"
 * and "a system that stays current".
 *
 * This is the closing moment of the demo: drop instruction 7.10.7 into the
 * bucket during a live demo, and the assistant can answer questions
 * about it about half a minute later (KR4: ≤ 60s).
 *
 * The handler is deliberately thin (AD-1). Everything it does lives in
 * core/ingest.ts, which is also what the CLI runs — so a bug found locally is
 * the same bug that was in the cloud, and fixing it fixes both.
 */

import { ingest } from "../core/ingest.js";
import { collectionFromKey } from "../core/store.js";
import { INDEX_PREFIX } from "../core/config.js";

/** The slice of the S3 event shape we actually rely on. */
interface S3EventRecord {
  s3: { bucket: { name: string }; object: { key: string; size?: number } };
}
interface S3Event {
  Records: S3EventRecord[];
}

export const handler = async (event: S3Event): Promise<{ ingested: number; failed: number }> => {
  const records = event.Records ?? [];
  let ingested = 0;
  let failed = 0;

  for (const record of records) {
    // S3 percent-encodes keys in event notifications and turns spaces into "+".
    // Skipping this decode is a classic silent failure: the object exists but
    // the key we ask for does not.
    const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));

    // The function writes its own output back into the same bucket, so without
    // this guard an index write would re-trigger the function that wrote it —
    // an infinite, billable loop. The deployed event notification is also
    // filtered by prefix; this is the belt to that pair of braces.
    if (key.startsWith(INDEX_PREFIX)) {
      console.log(`דילוג על ${key} — זהו קובץ אינדקס, לא מסמך מקור`);
      continue;
    }

    const collection = collectionFromKey(key);

    try {
      const result = await ingest({
        collection,
        keys: [key],
        onLog: line => console.log(line),
      });

      ingested++;
      console.log(
        JSON.stringify({
          level: "info",
          event: "ingested",
          key,
          collection,
          chunks: result.chunks,
          embedded: result.embedded,
          tokens: result.tokens,
          costUSD: Number(result.costUSD.toFixed(6)),
          elapsedMs: result.elapsedMs,
        }),
      );
    } catch (err: any) {
      // One bad document must not abandon the rest of the batch (FR-1.4). The
      // failure is logged as structured JSON so CloudWatch Logs Insights can
      // query it, and the handler resolves rather than throwing — a throw would
      // make S3 retry the same broken object until it gave up.
      failed++;
      console.error(
        JSON.stringify({
          level: "error",
          event: "ingest_failed",
          key,
          collection,
          error: err?.name ?? "Error",
          message: err?.message ?? String(err),
        }),
      );
    }
  }

  return { ingested, failed };
};
