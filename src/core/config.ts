/**
 * config.ts — every model, region, bucket and switch in one place. (AD-9)
 *
 * The rule this file enforces: NO region string, model id or bucket name may
 * appear anywhere else in the codebase. That rule was learned the hard way —
 * the previous version had `new S3Client({region:"us-east-1"})` buried in the
 * ingest script while the config said us-west-2, and nothing pointed at the
 * disagreement until a call failed.
 *
 * The transport switches (AD-2) live here too. Quotas, availability and pricing
 * differ per account, per region and per model, and we have already been burnt
 * by all three. Code that hardcodes a provider gets rewritten when the provider
 * blocks you; code that reads one env var gets a one-value edit.
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Environment loading
// ---------------------------------------------------------------------------

/**
 * Load KEY=VALUE files into process.env without a dotenv dependency.
 * Two locations are checked, nearest-first: `poc/.env` (this POC's own secrets)
 * and the repository-root `.env.local` (where OPENROUTER_API_KEY already lives).
 * Existing environment variables always win, so Lambda's real configuration is
 * never overwritten by a stray file — in the cloud neither file exists at all.
 */
function loadEnvFiles(): void {
  const here = dirname(fileURLToPath(import.meta.url));
  const pocRoot = resolve(here, "..", "..");
  const candidates = [
    join(pocRoot, ".env"),
    join(pocRoot, ".env.local"),
    resolve(pocRoot, "..", ".env.local"),
  ];

  for (const file of candidates) {
    if (!existsSync(file)) continue;
    for (const line of readFileSync(file, "utf-8").split(/\r?\n/)) {
      const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
      if (!match) continue;
      const key = match[1];
      if (process.env[key] !== undefined) continue;
      process.env[key] = match[2].trim().replace(/^["']|["']$/g, "");
    }
  }
}

// Lambda sets AWS_LAMBDA_FUNCTION_NAME; there is no .env file to read there.
if (!process.env.AWS_LAMBDA_FUNCTION_NAME) loadEnvFiles();

const env = (name: string, fallback: string): string => process.env[name] ?? fallback;
const envNum = (name: string, fallback: number): number => Number(process.env[name] ?? fallback);

// ---------------------------------------------------------------------------
// Run mode
// ---------------------------------------------------------------------------

/**
 * "aws"   — documents and indexes live in S3 (the serverless deployment).
 * "local" — the same core logic against the local disk, so the demo survives a
 *           dead network. Not a mock: identical modules, different storage.
 */
export const RUN_MODE = env("RUN_MODE", process.env.AWS_LAMBDA_FUNCTION_NAME ? "aws" : "local") as
  | "aws"
  | "local";

/** Where local-mode artefacts live, relative to the poc/ folder. */
export const LOCAL_DATA_DIR = env("LOCAL_DATA_DIR", "data");
export const CORPUS_DIR = env("CORPUS_DIR", "corpus");

// ---------------------------------------------------------------------------
// S3 (AD-3)
// ---------------------------------------------------------------------------

/** Only used when RUN_MODE=aws. Set S3_BUCKET to a bucket with AES256 and public access blocked. */
export const S3_BUCKET = env("S3_BUCKET", "");
export const S3_REGION = env("S3_REGION", "us-east-1");

/** Documents: docs/{collection}/…  Indexes: index/{collection}.json */
export const DOCS_PREFIX = env("DOCS_PREFIX", "docs/");
export const INDEX_PREFIX = env("INDEX_PREFIX", "index/");

/**
 * Where the 18 chapter-7 files were originally uploaded, before the
 * per-collection layout existed. Kept so the first ingest can read them without
 * a migration; new uploads go to the docs/{collection}/ layout.
 */
export const LEGACY_CORPUS_PREFIX = env("LEGACY_CORPUS_PREFIX", "corpus/");

export const docsPrefixFor = (collection: string): string => `${DOCS_PREFIX}${collection}/`;
export const indexKeyFor = (collection: string): string => `${INDEX_PREFIX}${collection}.json`;

/**
 * Above this many chunks, an index stops storing its vectors inside the JSON
 * and moves them to a companion binary file. (FR-4.3 / D-1)
 *
 * The number is set from a measurement, not from taste. TAKAM is 545 chunks and
 * 7.5MB of JSON — small enough that JSON costs nothing and is worth keeping for
 * how readable it is. The pilot over the Bedrock user guide projects 1,092 pages
 * → ~6,534 chunks → ~90MB, and that 90MB is ~6.7 million floats written as
 * decimal TEXT: every cold Lambda start would have to parse all of them back
 * into numbers before answering the first question. The same vectors as raw
 * little-endian Float32 are ~27MB and need no parsing at all.
 *
 * 2000 sits in the empty space between those two measured points (roughly 27MB
 * of JSON): comfortably above anything the TAKAM demo will ever produce, so the
 * verified demo path is untouched, and comfortably below the first collection
 * that actually hurts. It is intentionally NOT a tight bound — a threshold that
 * flips format on a corpus growing by one document is a threshold that makes
 * behaviour hard to predict.
 *
 * See store.ts for the format itself and for INDEX_FORMAT, the override that
 * forces the choice either way when testing.
 */
export const VECTOR_BINARY_THRESHOLD = envNum("VECTOR_BINARY_THRESHOLD", 2000);

// ---------------------------------------------------------------------------
// Collections (AD-4)
// ---------------------------------------------------------------------------

export interface CollectionDef {
  id: string;
  /** Shown in the UI picker. */
  label: string;
  language: "he" | "en";
  /** How a citation reads: TAKAM cites an instruction, AWS cites a doc page. */
  citation: "instruction" | "page";
  /**
   * Set only on a VIRTUAL collection — one that has no index of its own and is
   * answered by searching several real collections and merging the results.
   *
   * It exists because picking the corpus is a worse job for the user than for
   * the system: to ask "what is a cold start and how do I reduce it" you would
   * otherwise have to already know that cold starts are a Lambda topic. The
   * similarity scores decide better than a guess does.
   *
   * A virtual collection is available only when every member it names has a
   * built index, which is why availability is checked through
   * `collectionAvailable` rather than by looking for one index file.
   */
  members?: string[];
}

/** The default collection — the one the demo opens on. */
/**
 * The collection a question goes to when nobody picks one.
 *
 * Changed from "takam" to "takam-all" on 02/09/2026, and the reason is the
 * single largest defect found that day. "takam" is the ORIGINAL demo corpus:
 * 545 chunks, 18 instructions, 17 of them from chapter 7. It was the whole
 * corpus once. When the other 393 instructions arrived on 30/08 they were added
 * as 15 new per-chapter collections ALONGSIDE it, and this line did not move.
 *
 * So the shipped default quietly searched 18 documents out of 411. A question
 * about annual leave, government vehicles or grants could not be answered — not
 * because retrieval was weak, but because the instruction that answers it was
 * never in the search. From the outside that is indistinguishable from a bad
 * model, which is exactly how it was reported: "it missed answers that are in
 * the TAKAM".
 *
 * "takam" stays in the picker. Its threshold was calibrated on precisely those
 * 18 documents and the demo runs against it, so it keeps its place —
 * as a deliberate choice rather than as the accidental default.
 */
export const DEFAULT_COLLECTION = env("DEFAULT_COLLECTION", "takam");

/**
 * The original 18-document demo corpus, as a REAL collection id.
 *
 * It exists as its own constant because several places need "the concrete
 * TAKAM collection" and used to spell that `DEFAULT_COLLECTION`. That was the
 * same string until 02/09/2026 and silently stopped being it: the legacy
 * `corpus/` shim, the flat-folder fallback in listDocumentKeys, and the
 * S3-event router all quietly re-pointed themselves at a VIRTUAL collection
 * that can hold no documents at all. Naming the two ideas separately is what
 * stops the next default change from doing the same thing again.
 */
export const LEGACY_COLLECTION = "takam";

/**
 * The AWS guides mirrored locally under `aws-docs/` (135 guides, 37,457 pages,
 * expanded 30/08/2026 by `_tools/launch-parallel.mjs`). Listing a service here does
 * NOT make it appear in the demo: both `/api/collections` handlers filter this
 * list down to the collections that actually have an index built, so a service
 * that was downloaded but never embedded stays invisible until it is. That is
 * deliberate — the archive is a reading source for us, the index is what the
 * demo answers from, and the two are allowed to differ.
 *
 * Display name first, slug second, because `aws-bedrock-agentcore` reads badly
 * in a picker and the label is the only thing a person in the room sees.
 */
export const AWS_DOC_SERVICES: ReadonlyArray<readonly [slug: string, label: string]> = [
  // Core — the services this POC is actually built from.
  ["bedrock", "Bedrock"], ["bedrock-agentcore", "Bedrock AgentCore"],
  ["opensearch", "OpenSearch"], ["lambda", "Lambda"], ["s3", "S3"],
  ["iam", "IAM"], ["apigateway", "API Gateway"], ["textract", "Textract"],
  ["kendra", "Kendra"],
  // Infrastructure and operations.
  ["cloudwatch", "CloudWatch"], ["cloudformation", "CloudFormation"],
  ["vpc", "VPC"], ["kms", "KMS"], ["secretsmanager", "Secrets Manager"],
  ["cognito", "Cognito"], ["eventbridge", "EventBridge"],
  ["stepfunctions", "Step Functions"], ["sqs", "SQS"], ["ecs", "ECS"],
  ["cdk", "CDK"], ["cli", "CLI"], ["sdk-js", "SDK for JavaScript"],
  // Data.
  ["dynamodb", "DynamoDB"], ["aurora", "Aurora / RDS"], ["glue", "Glue"],
  ["athena", "Athena"], ["glacier", "S3 Glacier"],
  // Governance, security and cost.
  ["config", "Config"], ["cloudtrail", "CloudTrail"],
  ["organizations", "Organizations"], ["controltower", "Control Tower"],
  ["cost-management", "Cost Management"],
  ["wellarchitected", "Well-Architected"], ["general", "General Reference"],
  ["sra", "Security Reference Architecture"], ["aws-overview", "AWS Overview"],
  // Other AI services.
  ["comprehend", "Comprehend"], ["transcribe", "Transcribe"],
  ["polly", "Polly"], ["translate", "Translate"],
  ["rekognition", "Rekognition"], ["sagemaker", "SageMaker"],
  // Added 30/08/2026 with the archive expansion. The first four are the ones
  // that matter most for this POC's subject matter: `prescriptive-guidance` is
  // AWS's own written guidance on building exactly this kind of system (how to
  // choose a vector database for RAG, how to secure a generative-AI workload),
  // and `amazonq` is the managed enterprise-RAG product this POC is the
  // hand-built counterpart to — being able to cite both is the difference
  // between "I built a RAG" and "I built a RAG and know where it sits".
  ["prescriptive-guidance", "Prescriptive Guidance (GenAI)"],
  ["whitepapers", "Whitepapers"],
  ["certification", "Certification Exam Guides"],
  ["amazonq", "Amazon Q"], ["nova", "Nova"],
  // Downloaded and available to index, listed so the picker can grow without a
  // code change. They stay invisible until an index is actually built.
  ["eks", "EKS"], ["rds", "RDS"], ["redshift", "Redshift"],
  ["guardduty", "GuardDuty"], ["securityhub", "Security Hub"],
  ["macie", "Macie"], ["inspector", "Inspector"], ["waf", "WAF"],
  ["network-firewall", "Network Firewall"], ["audit-manager", "Audit Manager"],
  ["detective", "Detective"], ["singlesignon", "IAM Identity Center"],
  ["acm", "Certificate Manager"], ["ram", "Resource Access Manager"],
  ["batch", "Batch"], ["apprunner", "App Runner"], ["sam", "SAM"],
  ["systems-manager", "Systems Manager"], ["lake-formation", "Lake Formation"],
  ["emr", "EMR"], ["msk", "MSK"], ["firehose", "Data Firehose"],
  ["kinesis", "Kinesis Data Streams"], ["elasticache", "ElastiCache"],
  ["documentdb", "DocumentDB"], ["neptune", "Neptune"],
  ["memorydb", "MemoryDB"], ["quicksight", "QuickSight"],
  ["datazone", "DataZone"], ["databrew", "Glue DataBrew"],
  ["route53", "Route 53"], ["elb", "Load Balancing"],
  ["cloudfront", "CloudFront"], ["global-accelerator", "Global Accelerator"],
  ["efs", "EFS"], ["aws-backup", "Backup"], ["datasync", "DataSync"],
  ["transfer", "Transfer Family"], ["xray", "X-Ray"], ["sns", "SNS"],
  ["amazon-mq", "MQ"], ["mwaa", "MWAA (Airflow)"],
  ["codepipeline", "CodePipeline"], ["codebuild", "CodeBuild"],
  ["codedeploy", "CodeDeploy"], ["amplify", "Amplify"],
  ["appsync", "AppSync"], ["servicecatalog", "Service Catalog"],
  ["health", "Health"], ["personalize", "Personalize"],
  ["frauddetector", "Fraud Detector"], ["lexv2", "Lex V2"],
  ["sagemaker-unified-studio", "SageMaker Unified Studio"],
] as const;

/**
 * The TAKAM main chapters, from the ministry's own /api/Chapter endpoint
 * (fetched 30/08/2026). Names are theirs, not ours — a picker that renames a
 * government chapter is a picker that disagrees with the source it cites.
 *
 * Splitting by chapter rather than keeping one big `takam` index is the same
 * decision, and for the same reason, as splitting AWS by service: retrieval
 * here is a linear cosine scan (AD-7), so every chunk in the pool costs time on
 * every question AND is another chance for an irrelevant passage to score high.
 * The full instruction set is ~22x the 545 chunks the demo was calibrated on;
 * pouring that into one collection would slow down the chapter-7 questions the
 * demo already answers well, to buy nothing.
 *
 * Chapter 10 is absent from the ministry's own list and chapter 16 carries
 * tender notices rather than instructions — both are kept here anyway, because
 * documents do map to them and an unlisted chapter would silently drop content.
 */
export const TAKAM_CHAPTERS: ReadonlyArray<readonly [code: string, label: string]> = [
  ["1", "ביצוע תקציב"],
  ["2", "חשבונאות ממשלתית"],
  ["3", "ריבית הכנסות וגבייה"],
  ["4", "כיסוי ביטוחי וטיפול בתביעות"],
  ["5", "ניהול סיכונים וביקורת"],
  ["6", "תמיכות"],
  ["7", "התקשרויות ורכישות"],
  ["8", "נותני שירותים חיצוניים ועובדי קבלן"],
  ["9", "עודפי טובין ובלאי"],
  ["10", "רכש ולוגיסטיקה"],
  ["11", "רכב ממשלתי"],
  ["12", "דיור ממשלתי"],
  ["13", "שכר, תנאי שירות וגמלאות"],
  ["14", 'שכר ותנאי שירות - חו"ל'],
  ["15", "שונות"],
  ["16", "הודעות מכרזים מרכזיים"],
] as const;

export const COLLECTIONS: CollectionDef[] = [
  /**
   * The original 18-instruction chapter-7 corpus. Kept as its own collection,
   * and kept FIRST, because it is the one the demo runs against and
   * its retrieval threshold was calibrated on exactly these documents. The
   * per-chapter collections below are additive; nothing about the verified demo
   * path changes.
   */
  { id: "takam", label: 'הוראות תכ"ם — פרק 7 (15 הוראות)', language: "he", citation: "instruction" },
  ...TAKAM_CHAPTERS.map(([code, label]): CollectionDef => ({
    id: `takam-${code}`,
    label: `תכ"ם ${code} — ${label}`,
    language: "he",
    citation: "instruction",
  })),
  ...AWS_DOC_SERVICES.map(([slug, label]): CollectionDef => ({
    id: `aws-${slug}`,
    label: `AWS ${label}`,
    language: "en",
    citation: "page",
  })),
];

/**
 * The virtual collection that searches every AWS guide at once.
 *
 * Placed AFTER the real collections so it can be derived from them rather than
 * repeating the list — adding a service to AWS_DOC_SERVICES puts it in here for
 * free, and a hand-maintained second list is a list that eventually disagrees
 * with the first.
 *
 * It is unshifted to position 1 (right after TAKAM) because it is the option a
 * person actually wants: "search the AWS documentation" is the question, and
 * "which of nine services should I search" is an implementation detail they
 * should not have to answer first.
 */
COLLECTIONS.splice(1, 0, {
  id: "aws-all",
  label: "כל מדריכי AWS",
  language: "en",
  citation: "page",
  members: COLLECTIONS.filter(c => c.id.startsWith("aws-")).map(c => c.id),
});

/**
 * The TAKAM counterpart to `aws-all`. Same reasoning: "search the TAKAM" is the
 * question a person has; "which of sixteen chapters" is an implementation
 * detail they should not have to answer. Members are derived from the chapter
 * list so adding a chapter joins it for free.
 *
 * `takam` itself is deliberately NOT a member — it is a subset of chapter 7 and
 * including it would return the same instruction twice under two ids.
 */
COLLECTIONS.splice(1, 0, {
  id: "takam-all",
  label: 'כל הוראות התכ"ם',
  language: "he",
  citation: "instruction",
  members: COLLECTIONS.filter(c => c.id.startsWith("takam-")).map(c => c.id),
});

export const collectionDef = (id: string): CollectionDef =>
  COLLECTIONS.find(c => c.id === id) ?? COLLECTIONS[0];

// ---------------------------------------------------------------------------
// Embeddings (AD-2) — runs on real AWS
// ---------------------------------------------------------------------------

/**
 * Re-verified 30/08/2026: the per-model quotas opened. EVERY embedding model in
 * the account now answers — Titan v1/v2, Cohere v3 (multilingual + english),
 * Cohere v4. The 21/08 picture (only Cohere v3 worked) is obsolete.
 *
 * The default moved from Cohere v3 to Titan v2 because it was MEASURED, not
 * assumed: Titan separates a relevant Hebrew instruction from an irrelevant one
 * 3.5x better (0.393 vs 0.111 average gap over four query pairs), costs 5x less,
 * and returns the same 1024 dims so nothing downstream changes shape.
 * The measurement and its consequence for MIN_SCORE are documented on
 * `titanTransport` in embed.ts and on MIN_SCORE below.
 *
 * Bedrock quotas are PER MODEL — one blocked model never means the service is
 * closed, and one OPEN model never means the rest opened either.
 */
export const EMBED_TRANSPORT = env("EMBED_TRANSPORT", "bedrock:titan") as
  | "bedrock:cohere"
  | "bedrock:titan"
  | "gemini";

export const EMBED_REGION = env("EMBED_REGION", "us-east-1");

/**
 * Extra regions to spread embedding calls across (AD-2, 30/08/2026).
 *
 * The reason this exists is a measured limit, not a preference. Titan takes ONE
 * text per call and this account is capped at ~60 calls/minute PER REGION, so a
 * 38,000-chunk re-ingest is ~17 hours from us-east-1 alone. Adding workers does
 * nothing — the ceiling is the region's, not the client's, which is why raising
 * EMBED_CONCURRENCY made it slower rather than faster.
 *
 * Bedrock quotas are per model AND per region, so five regions are five separate
 * ceilings. Verified 30/08/2026 by invoking Titan in each and CHECKING THE
 * RETURNED VECTOR — the first sweep of this list counted an empty response as
 * success and reported eleven regions, six of which answer
 * "ValidationException: The provided model identifier is invalid". A probe that
 * cannot fail is not a probe; assert on the payload, not on the absence of noise.
 *
 * The correctness question this raises is whether a vector from Frankfurt is
 * comparable to one from Virginia. It is: the same input returned a
 * BYTE-IDENTICAL embedding from us-east-1, us-west-2 and eu-central-1
 * (cosine 1.00000000). Same model weights, different door. Had that not held,
 * this would be a corpus-corrupting idea rather than a scheduling one.
 *
 * RUN ONE INGEST AT A TIME. These five ceilings are shared by every process on
 * the account, not held per process. Running a second ingest "to catch up"
 * halves both and makes both fail: on 30/08/2026 that killed `aws-lambda` and
 * `aws-bedrock-agentcore` simultaneously, after each had run for ten minutes.
 * The retry budget (8 attempts, ~2 minutes) is sized for one process draining
 * five regions; two processes exhaust it before either finishes.
 *
 * Failure here is safe but wasteful: a failed ingest writes nothing, so the
 * previous index survives intact and keeps answering through its own recorded
 * transport. Nothing corrupts — the run is simply lost.
 *
 * Data residency note: this moves TAKAM text through EU regions. That is
 * acceptable for a POC over PUBLIC published instructions, and would need
 * revisiting for anything non-public — set this to the empty string to pin all
 * embedding traffic back to EMBED_REGION.
 */
export const EMBED_REGIONS: string[] = env(
  "EMBED_REGIONS",
  "us-east-1,us-east-2,us-west-2,eu-central-1,eu-west-1",
).split(",").map(r => r.trim()).filter(Boolean);

/**
 * `pricePer1M` is USD per 1M input tokens, and it lives here PER MODEL rather
 * than as one shared constant — because the shared constant went wrong the
 * moment the default changed. It read $0.10, which is Cohere's price, while
 * every embedding in this POC has been produced by Titan at $0.02. Every cost
 * readout was therefore 5x the real figure. Over-reporting is the safe
 * direction to be wrong in, which is exactly why it survived so long: nothing
 * ever looked alarming enough to make anyone check it.
 */
export const EMBED_MODELS: Record<string, { modelId: string; dims: number; maxBatch: number; pricePer1M: number }> = {
  "bedrock:cohere": { modelId: "cohere.embed-multilingual-v3", dims: 1024, maxBatch: 96, pricePer1M: 0.1 },
  // maxBatch here is how many texts the TRANSPORT accepts per call, not how many
  // the API accepts. Titan's API still takes exactly one text per request; the
  // transport fans a batch out across EMBED_CONCURRENCY in-flight calls. Leaving
  // this at 1 (as it was while Titan was only a stub) makes embedAll hand over one
  // text at a time and pay EMBED_PACE_MS between each — 11 hours for this corpus.
  "bedrock:titan": { modelId: "amazon.titan-embed-text-v2:0", dims: 1024, maxBatch: 64, pricePer1M: 0.02 },
  gemini: { modelId: "gemini-embedding-001", dims: 1536, maxBatch: 100, pricePer1M: 0.15 },
};

/** How many chunks go up in one call. Cohere accepts 96; we stay a little under. */
export const EMBED_BATCH = envNum("EMBED_BATCH", 64);

/** Pause between batches — politeness toward a rate limit we have already hit. */
export const EMBED_PACE_MS = envNum("EMBED_PACE_MS", 250);

/**
 * How many single-text embed calls may be in flight at once (AD-2, 30/08/2026).
 *
 * Only transports with `maxBatch: 1` use this — today that is Titan, which takes
 * one text per call. Cohere batches 96 texts into a single request and therefore
 * ignores this entirely.
 *
 * Why it exists: the corpus is ~39,000 chunks, and one-at-a-time is roughly six
 * hours of wall clock.
 *
 * The history of this number is the useful part, because it reversed:
 *
 *  - 16, single region → SLOWER than 6. Titan throttled, the retry loop backed
 *    off 1-2-4-8-16-32-64s, and throughput fell below the serial rate. Against
 *    ONE ceiling, extra workers buy nothing and pay the backoff.
 *  - 6, single region → ~37 calls/min measured. ~17 hours for the AWS corpus.
 *  - 12, across the 5 EMBED_REGIONS → five ceilings instead of one, so workers
 *    stop competing for one and the number can rise.
 *
 * The rule that survives both: concurrency helps only up to the number of
 * INDEPENDENT rate ceilings available. Roughly two in flight per region is the
 * shape; raising this without adding regions just recreates the 16 case.
 */
export const EMBED_CONCURRENCY = envNum("EMBED_CONCURRENCY", 12);

// ---------------------------------------------------------------------------
// Generation (AD-2) — temporarily outside AWS, by quota not by choice
// ---------------------------------------------------------------------------

/**
 * 30/08/2026 — THE QUOTA OPENED. Generation runs on Bedrock, inside AWS.
 *
 * This is what the transport switch was built for: the day the block lifted, the
 * move was this one default, not a rewrite. Verified end to end, not by listing —
 * `LLM_ROUTE=bedrock:claude npm run ask` answered in Hebrew with citations.
 *
 * Bedrock Haiku 4.5 is ~10% dearer than the same model on OpenRouter
 * ($1.10/$5.50 vs $1.00/$5.00 per 1M). That is deliberate: the ministry's own
 * SDLC document names its provider as "Anthropic via AWS", so 10% on fractions of
 * a cent buys running exactly the stack the division runs.
 *
 * Still open at the account level (NOT quota — `create-foundation-model-agreement`
 * succeeds and the invoke is still refused with "not available for this account",
 * which is an AWS Sales matter): Opus 4.7/4.8, Fable 5, Sonnet 5, Opus 5, GPT-5.6.
 * Available if ever needed: Sonnet 4.5/4.6 ($3/$15), Opus 4.5/4.6 ($5/$25).
 *
 * The cheaper non-Anthropic models on Bedrock (GLM-4.7-flash at $0.07/$0.40,
 * DeepSeek V3.2 at $0.62/$1.85) are configured but NOT the default, and the
 * reason is not technical: a Ministry of Finance POC answering from a Chinese
 * model is a regulatory conversation, not a cost decision.
 */
export const LLM_ROUTE = env("LLM_ROUTE", "bedrock:claude") as
  | "openrouter"
  | "openrouter:claude"
  | "bedrock:deepseek"
  | "bedrock:claude"
  | "anthropic";

export const GEN_REGION = env("GEN_REGION", "us-east-1");

/**
 * GEN_REGIONS — spread generation calls over several regions, because the rate
 * ceiling is counted per region.
 *
 * THE PROBLEM THIS ADDRESSES. The gate metric is "first token within 6 seconds"
 * and the 15/09 gate run measured p90 = 9.1s. The retry meter (19/09) split that
 * number and showed the system is not slow: the 17 rows of a 40-question run
 * that were never throttled came in at p90 5.10s, under target, while 23 of 40
 * waited before their first word and 41 of 80 questions were rejected outright
 * and retried a level up. The ceiling is the account's, not the prompt's.
 *
 * WHAT WAS MEASURED, 20/09/2026 (eval/benches/2026-09-20-gen-regions/):
 *
 *  1. `probe-regions.mts` — one real Converse call per region, because a model
 *     agreement and a listing are not access. Five regions answer: us-east-1,
 *     us-east-2, us-west-2 through the `us.` inference profile, and
 *     eu-central-1, eu-west-1 through `eu.`. The `apac.` profile returns "model
 *     identifier is invalid" for Haiku 4.5. A `global.` profile also answers
 *     from us-east-1 — a different lever, noted and not used here.
 *     In the same minute, eu-west-3 and eu-north-1 refused with "Too many
 *     tokens PER DAY" while eu-central-1 and eu-west-1 served normally. So the
 *     DAILY budget is bookkept per region too, which is worth knowing on a day
 *     when us-east-1 says the day is over.
 *
 *  2. The shape of the ceiling, learned from two probes that failed honestly:
 *     8 calls fired at once against one region throttle nothing (so it is not a
 *     burst limit), and a region that has just refused twice answers again two
 *     seconds later (so it refills in seconds). It is a rolling window: ~18,000
 *     tokens in four seconds is fine, ~40,000 over a minute is not.
 *
 *  3. `hammer-and-probe.mts` — the discriminator, and the only one of the four
 *     that could have come out the other way. Three workers hammering us-east-1
 *     continuously, and the instant a call was refused, one call each to
 *     us-west-2 and eu-central-1 WITHOUT pausing the hammer. Result: us-east-1
 *     13/29, sixteen ThrottlingExceptions; the other two regions 6/6 in the very
 *     same seconds. Separate ceilings, measured rather than inferred from AWS
 *     documentation.
 *
 * WHY THIS IS A LIST AND NOT A RETRY TWEAK. The embedding side answered the
 * identical question on 30/08/2026 and the lesson transferred intact: against
 * ONE ceiling, more attempts buy nothing and pay the backoff; against five
 * ceilings, the same traffic simply fits. EMBED_CONCURRENCY documents the
 * reversal in full a few hundred lines above.
 *
 * DEFAULT OFF (iron rule 10). Empty means "pin everything to GEN_REGION", which
 * is exactly today's behaviour, byte for byte. Turning it on is Elad's call:
 *
 *     GEN_REGIONS=us-east-1,us-east-2,us-west-2                 (US only)
 *     GEN_REGIONS=us-east-1,us-east-2,us-west-2,eu-central-1,eu-west-1
 *
 * Data residency, the same note EMBED_REGIONS carries: the second list moves
 * TAKAM text through EU regions. Acceptable for a POC over PUBLIC published
 * instructions, and a conversation to have before anything non-public. The US
 * list keeps three ceilings without leaving the geography the model already
 * runs in — `us.` is itself a cross-region profile that routes inside the US.
 *
 * APPLIES TO `bedrock:claude` ONLY. Rotation needs a geography-prefixed
 * inference profile to rewrite; `bedrock:deepseek` is a plain ON_DEMAND model in
 * us-east-1 and any other region would be a ValidationException, so it stays
 * pinned to GEN_REGION whatever this says. See `regionalModelId` in answer.ts.
 */
export const GEN_REGIONS: string[] = env("GEN_REGIONS", "")
  .split(",").map(r => r.trim()).filter(Boolean);

/**
 * Price per 1M tokens, for the cost line shown under every answer.
 *
 * The pairing is the point: every route has a counterpart that runs the SAME
 * model family inside AWS. Opening the quota does not change the product, the
 * prompt or the code — it changes this one value.
 *
 *   openrouter:claude  ⟷  bedrock:claude     (Claude Haiku 4.5)
 *   openrouter         ⟷  bedrock:deepseek   (DeepSeek V3.2)
 *
 * Default is `openrouter:claude` because the ministry's own SDLC document names
 * its LLM provider as "Anthropic via AWS" — so the model that should be running
 * here is Claude, and the only reason it is not running ON Bedrock is the quota.
 * DeepSeek stays configured as the cheaper alternative (roughly a quarter of the
 * price) and as proof the transport really is model-agnostic.
 */
export const GEN_MODELS: Record<string, { modelId: string; priceIn: number; priceOut: number }> = {
  "openrouter:claude": { modelId: "anthropic/claude-haiku-4.5", priceIn: 1, priceOut: 5 },
  openrouter: { modelId: "deepseek/deepseek-v3.2", priceIn: 0.269, priceOut: 0.4 },
  // ON_DEMAND in us-east-1, no inference profile needed (verified 21/08/2026).
  "bedrock:deepseek": { modelId: "deepseek.v3.2", priceIn: 0.28, priceOut: 0.42 },
  "bedrock:claude": { modelId: "us.anthropic.claude-haiku-4-5-20251001-v1:0", priceIn: 1, priceOut: 5 },
  anthropic: { modelId: "claude-haiku-4-5-20251001", priceIn: 1, priceOut: 5 },
};

/**
 * The answer's output budget, in tokens.
 *
 * Raised from 900 on 02/09/2026. 900 was almost certainly copied from
 * CHUNK_CHARS just below — the two are unrelated quantities, and unlike every
 * other tuned constant in this file it carried no measurement behind it.
 *
 * Why 900 was actively harmful here. Hebrew tokenises badly: roughly 2.5
 * characters per token against 4 for English, so 900 tokens is about 2,200
 * Hebrew characters — some 350 words. Meanwhile the INPUT grew: TOP_K=5 hits,
 * each widened by NEIGHBOUR_RADIUS into a trio, is up to 15 passages of 900
 * characters, about 13,500 characters of regulation. The model was being handed
 * six times more than it was allowed to answer with.
 *
 * A real TAKAM answer has to state the rule, its threshold, its exceptions, who
 * approves it, and cite each. That does not fit in 350 words, so the model
 * stopped mid-answer — and because no transport read the stop reason, a
 * truncated answer came back looking exactly like a complete one. That is the
 * most likely mechanism behind "the bot missed answers that were inside the
 * instructions it read": it had not missed them, it had run out of room.
 *
 * 2500 tokens is ~6,000 Hebrew characters, enough for a full answer with
 * citations and still a small fraction of the model's window. Output tokens on
 * Haiku cost more than input, so this is not free — but at these volumes the
 * difference is fractions of a cent per question, and a truncated answer is
 * worth nothing at all.
 *
 * Raised to 8000 on 02/10/2026. "When may we contract without a tender, and who
 * approves?" is answered from 7.6.1's approval table, which lists some thirty
 * regulation sub-clauses, each with its own approver and threshold. At 2500 the
 * answer stopped mid-word, and the UI's first example question still stopped at
 * 4000 (4,012 tokens out). Haiku 4.5 allows far more output. The cap costs
 * nothing on answers that do not reach it: the longest answer in the 36-question
 * eval was 844 characters, about 600 tokens. Answer.truncated reports a cut-off,
 * so the cap is visible to the reader if it is ever hit again.
 */
export const GEN_MAX_TOKENS = envNum("GEN_MAX_TOKENS", 8000);

/**
 * How long an answer to write: "full" (default) or "concise".
 *
 * This is a SPEED control, and it is the only one that actually exists.
 * Measured 03/09/2026 on three demo questions: of 11.6 seconds, 10.5 were the
 * model writing and ~1 was retrieval. Nothing about retrieval — top-k, the
 * index format, a different embedding model — can move a number that is not
 * where the time is. The only two levers on generation are streaming (shipped,
 * and it changes nothing about the answer) and writing less.
 *
 * Measured cost and benefit of "concise", same passages, same model:
 *
 *   question                    full          concise       speed-up
 *   פטור ממכרז                  11.9s / 915   10.1s / 607   1.2x
 *   רכישת שירותי AI             20.4s / 1439  11.5s / 773   1.8x
 *   זכויות עובדים               21.0s / 1465   8.3s / 478   2.5x
 *
 * So it roughly halves the wait on a long answer. It is NOT the default, and
 * the reason is a measured regression rather than caution: an earlier, blunter
 * wording ("up to 200 words, four bullets") dropped instruction 7.1.1 from the
 * פטור-ממכרז answer — the half that names the approving authority. The wording
 * below fixes that by saying that shortening means less explanation and never
 * fewer sources, and it does recover 7.1.1. But on the employee-rights question
 * it made the model open with a hedge before answering, which in a live demo
 * reads as a failure even though the citation is still there.
 *
 * A default that is faster on two questions and reads worse on a third is not a
 * default. It is an option, so it is an option: set ANSWER_STYLE=concise.
 */
export const ANSWER_STYLE = env("ANSWER_STYLE", "full") === "concise" ? "concise" : "full";

/**
 * D-3 (weighting exact keyword matches in the retrieval score) was IMPLEMENTED
 * and then REVERTED on 03/09/2026. The attempt and its measurement are recorded
 * here because the negative result is the useful part.
 *
 * Trigger: "מה התעריף של בודק תוכנה" refused while the rate (99 ₪) sat in the
 * corpus in 17 chunks — inside a bare rate TABLE, rows of role names and
 * numbers. A table has no semantic handle, so no embedding can reach it.
 *
 * Two designs were built and measured against the twelve-question battery:
 *   1. An additive boost (+0.1 for a rare exact phrase). Measured: no effect.
 *      The table sits near 0.3 semantically because its embedding is mostly
 *      numbers, so the boost left it under the threshold.
 *   2. Promotion — a rare phrase lifts the passage to just over the refusal
 *      threshold, as a second retriever whose results join the first.
 *      Measured, and this is the reason for the revert: THREE control questions
 *      that must refuse stopped refusing. "כמה מס הכנסה משלם שכיר" went from
 *      0.415 to 0.500, "פיטורי עובד בחברה פרטית" and "ביטוח לאומי" likewise —
 *      all promoted on incidental word pairs. The target question still failed,
 *      because a document-frequency ceiling expressed as a PERCENTAGE is too
 *      tight on a collection with few documents and many chunks: chapter 16
 *      holds 7,415 chunks in a few hundred documents, so the real signal
 *      ("בודק תוכנה", 5 documents) was rejected as "not rare enough" while
 *      one-off noise elsewhere qualified.
 *
 * So the change made the system worse at the property that matters most —
 * refusing precisely — and did not fix what it was built for. Reverted rather
 * than tuned, because tuning a threshold against three failures hours before it
 * has to be trusted is how a system stops being predictable.
 *
 * D-3 stays deferred. Doing it properly means a real hybrid retriever (BM25 or
 * equivalent, with rank fusion) and an evaluation set larger than twelve
 * questions — not a constant to pick.
 */
/** Appended to the system prompt when ANSWER_STYLE=concise. See above. */
export const CONCISE_STYLE_RULES = `

סגנון התשובה:
- כתוב תמציתי: עד כ-200 מילים, בלי כותרות משנה, בלי חזרות ובלי פסקת סיכום.
- אם השאלה מורכבת מכמה חלקים, ענה על כל חלק — גם אם התשובה חייבת להיות קצרה יותר כדי להספיק.
- אין לוותר על ציטוטים כדי לקצר. כל הוראה שהשתמשת בה חייבת להופיע בתשובה בשמה, גם אם היא תורמת משפט אחד.
- לקצר משמעו פחות הסבר, לא פחות מקורות.`;

/**
 * ANSWER_APPLICABILITY — let the model DERIVE an answer from a rule it holds,
 * instead of refusing because the corpus never phrases the question back.
 *
 * Measured on the 10/09 gate run, and this flag exists because of that
 * measurement rather than ahead of it. Of the 25 wrong answers, 6 are rows
 * where the correct instruction was ranked FIRST and the answer still opened
 * with "לא מצאתי במסמכים שברשותי". Reading those six, five of them go on to
 * quote the expected value verbatim in the next sentence. g-190 is the clearest:
 * asked whether a framework tender may be held as a closed tender, the model
 * answers "I did not find it", then quotes 7.3.5 — "מכרז מסגרת ייערך אך ורק
 * כמכרז פומבי" — and observes that this indicates a closed one is not possible,
 * "but the passages contain no explicit negative statement on the matter".
 *
 * That is not a retrieval failure and not a knowledge failure. It is the prompt
 * offering exactly two moves, answer or refuse, to a model that holds a rule the
 * answer FOLLOWS FROM but cannot find the answer pre-phrased. Faced with that,
 * it hedges: it files the disclaimer first and answers second.
 *
 * The cost of the hedge is not only cosmetic. `sources` is cleared for anything
 * whose text opens with the refusal sentence (see answer.ts), so all six rows
 * reach the judge with zero citations and cannot be scored correct no matter
 * what they say. That is the measured reason the 11/09 judge fix — which was
 * written precisely to credit a refusal that answers anyway — flipped zero rows.
 *
 * So the rule added here is about DERIVATION, and it is deliberately narrow:
 * a rule in the passages that settles the question is an answer to the question,
 * and a full refusal is reserved for having no basis at all. The second half
 * covers the other half of the same defect, found in the controls: two adjacent
 * controls answered from the contractors annex without saying the annex does not
 * apply to the asker. Partial applicability is to be stated, not silently
 * dropped and not escalated to a refusal.
 *
 * Kept OFF by default, and measured with it on, because it loosens the
 * condition under which the system agrees to answer — the one direction where
 * being wrong is expensive. The control set is what guards it.
 */
export const ANSWER_APPLICABILITY = env("ANSWER_APPLICABILITY", "0") !== "0";

/**
 * ANSWER_SCOPE_FIRST — when the passages' rule governs a narrower body than the
 * question asks about, the answer's FIRST sentence says whom it governs, and it
 * does not open with a bare yes/no or a number. Added 22/09/2026.
 *
 * WHY. The refusal target was split on 18/09/2026 into two definitions (91/100 by
 * the code's count, 98/100 by the stage-4 exit gate, which counts an answer that
 * states applicability as correct). Two controls fail under BOTH, and they are
 * the dangerous kind — a confident wrong headline, not a hedge:
 *   c-021  "כן, המעסיק חייב להפריש לפנסיה." + 7.5%/8.33% from annex 8.2.1, which
 *          binds service CONTRACTORS in government contracts. Opens "כן, (ה)מעסיק
 *          חייב להפריש לפנסיה" in all 11 results files that contain it (two
 *          variants, differing by the article only).
 *   c-098  "... הריבית המקסימלית ... היא 11.25%" for a first-home bank mortgage,
 *          from 3.1.1 — late-payment interest on STATE loans. The model knows: the
 *          caveat is in its own last paragraph. Same opening in 4/4 files.
 *
 * Measured before writing this (eval/benches/2026-09-22-scope-first/population.mts,
 * zero cost, saved answers of the 4-flag gate run): of the 9 controls that were not
 * refused, exactly these two open with yes/no or a number; the other seven already
 * open with the scope ("הקטעים עוסקים בדיור ממשלתי בלבד"). So the defect is ORDER,
 * not knowledge. ANSWER_APPLICABILITY's second half ("state what does not apply")
 * never ran on either of them — no results file with it on contains them.
 *
 * WHAT IT CAN BREAK, measured on the same file: 83 of 259 correct gold answers
 * open with yes/no or a number, without a scope phrase, to a question that does
 * not name the manual's frame — the rows the rule could rewrite. And 87 of the 91
 * correct control refusals are model-written, so a prompt rule reaches them; the
 * rule is therefore worded to govern only how an answer OPENS and to leave rule 2
 * (when to refuse) untouched. Loosening a refusal is the expensive direction; c-071
 * and c-076 (private-sector employee, contractors' annex in the passages — c-021's
 * exact shape, correctly refused today) are the adversarial guards for it.
 *
 * TAKAM collections only: the AWS prompt has no "body the rule binds".
 * Evidence per answer: Answer.scopeFirstRule (did the prompt actually sent carry
 * the rule). OFF by default; switching it on is Elad's decision, like every flag.
 *
 * MEASURED 22/09/2026 — NOT A CANDIDATE in this wording. With the 4 gate flags:
 *   9 targeted rows (eval/results/2026-09-21T22-13-16.json): c-098 now opens with
 *   "התשובה חלה על הלוואות מכספי המדינה"; c-021 carries the scope in its first
 *   sentence but still opens with "כן" ("...מתייחסות בעיקר ל... קבלני שירותים");
 *   c-071 c-076 c-017 stay refused; gold 4/4 correct.
 *   40+40 (eval/results/2026-09-21T22-27-38.json, valid, rule sent 76/76), paired
 *   with the 4-flag gate: controls refused 35 -> 37 with 0 refusal->answer flips;
 *   gold 22 -> 21. The cost and the gain have ONE mechanism, and it is not the
 *   rule's intent: the preamble "על סמך הקטעים המצורפים," disappeared (gold 7 -> 2,
 *   controls 4 -> 0), so a partial answer now STARTS with the refusal sentence and
 *   is counted as a refusal. c-009 and c-028 say nearly the same thing as before;
 *   g-023 also dropped the fact ("בתוך 30 ימים", present in all 10 results files
 *   from 10/09 to 19/09, text identical to the gate's in 8 of them) and lost its source. Exposed on the full gate, derived: 3 correct gold
 *   (g-023 g-046 g-132) open "על סמך הקטעים ... לא מצאתי", and 7 answered controls
 *   open with the preamble. The next wording must leave that opening alone.
 */
export const ANSWER_SCOPE_FIRST = env("ANSWER_SCOPE_FIRST", "0") !== "0";

/**
 * REFUSAL_KEEPS_QUOTED_SOURCES — an answer that opens with the refusal sentence
 * and then answers anyway keeps the citations it QUOTES, instead of all of them
 * being deleted.
 *
 * Measured 11/09/2026, and this one is not a hypothesis. Of the 6 rows in the
 * 10/09 gate run where the right instruction was ranked first and the answer
 * refused, 5 contain the expected value in their text. All 6 reach the judge
 * with zero sources, because answer.ts clears `sources` for anything whose text
 * opens with the refusal sentence. `correct` requires expectCited, expectCited
 * requires a citation, so those rows cannot be scored correct no matter what
 * they say. That is the measured reason the 11/09 judge fix, written precisely
 * to credit a refusal that answers anyway, flipped zero rows: the fix was right
 * and the evidence it needed had already been thrown away upstream.
 *
 * Derived from the paired subset run of 11/09 at zero cost: on 3 of 7 scorable
 * rows the model writes the expected instruction's own code and quotes its text,
 * so restoring those citations carries them from wrong to correct. That is
 * 3 of 7 here, and up to ~1.8 points of the 272-question denominator — larger
 * than the 1.6 points left to the 95% target. It is a DERIVATION from a valid
 * run, not a measured run result, and it is not official until a gate run.
 *
 * OFF by default. It widens what may appear under "מקורות" on a refusal, which
 * is the 08/09 bug's direction, so the gate run and the control set decide.
 *
 * The control set has now spoken, and it says NOT YET. Measured 14/09/2026 on
 * the first 40 gold + 40 controls with the flag on, eval/results/
 * 2026-09-14T05-47-54.json (valid, reranked 40/40): refusals 35/40, identical
 * to the same 40 ids in the 10/09 gate run — the flag cannot change who is
 * refused, as designed. But 7 of those 35 CORRECT refusals now carry a source
 * (c-001 c-004 c-016 c-017 c-019 c-031 c-034), against 0 with the flag off.
 * Each one quotes, in quotation marks, the passage it REJECTED: "income tax is
 * mentioned only for Foreign Ministry emissaries (HOD.14.7.8.1)". The quote test
 * separates "the model quoted something" from "the model did not"; it does not
 * separate "the quote answers" from "the quote explains why nothing answers".
 * The 11/09 estimate — 9 of 10 control refusals keep an empty box — was taken
 * on 8 controls; on 40 it is 28 of 35. Gold side on the same 40: one row gained
 * (g-027). So promoting the flag needs a sharper discriminator, not only a gate
 * run. The page renders the case honestly either way (public/index.html,
 * `.partial`), which is what makes the flag safe to try — not safe to default.
 */
export const REFUSAL_KEEPS_QUOTED_SOURCES = env("REFUSAL_KEEPS_QUOTED_SOURCES", "0") !== "0";

/**
 * REFUSAL_QUOTE_CHECK — the sharper discriminator REFUSAL_KEEPS_QUOTED_SOURCES
 * was waiting for. A refusal whose quote survives the quote test gets ONE short
 * closed model call that classifies the text after the refusal sentence as
 * "עונה" (it gives the question itself an answer, a rule it follows from, or a
 * negative answer) or "מסביר" (it explains the instructions concern another
 * case, body or context, or points elsewhere). Only "עונה" keeps the sources.
 * Meaningless unless REFUSAL_KEEPS_QUOTED_SOURCES is on. OFF by default.
 *
 * Derived 14/09/2026 at a few cents, on STORED answers only - no answer was
 * generated for this. The population is exactly the one the check will see:
 * refusals that pass the quote test. An offline bench re-ran the production
 * quotedSpans/isRefusalText on saved answers against the corpus, and first had
 * to reproduce the 14/09 run - it did, the same 8 rows the flag actually kept.
 *
 *   dev set, 10/09 gate run texts: 5 gold rows the sources would make correct
 *     (g-027 g-190 g-218 g-223 g-235) against 15 that must stay empty
 *     (the 13 controls c-001 c-004 c-016 c-017 c-019 c-031 c-034 c-044 c-060
 *     c-063 c-083 c-092 c-096, and g-052 g-177)
 *   quote alone, two wordings:        0/5 and 1/5 kept   - yes/no labels too, so
 *                                     confounded with the trap below; not re-tested
 *   full text, yes/no labels:         1/5 and 0/5 kept   - the trap below
 *   full text, עונה/מסביר (shipped):   3/5 kept, 0/15 wrongly kept, 0/20 swaps
 *     (each positive's text paired with another positive's question), and the
 *     same verdicts on a repeat run
 *   held out, other generations (11/09 subset runs + 14/09): 9/16 kept,
 *     0/19 wrongly kept
 *
 * The trap, and why the labels are not כן/לא: a one-word yes/no verdict about a
 * question that is ITSELF a yes/no question gets answered as the question.
 * "האם אפשר להאריך את תקופת הפיילוט?" came back "לא" four prompts running while
 * the text under it ends "לא ניתן להאריך את תקופת ההדגמה" - the model was
 * answering the pilot question, not classifying. A trivial sanity prompt said
 * "כן" correctly, so the transport was fine; renaming the labels fixed it.
 *
 * What it still misses, consistently: g-190, whose own text adds "הקטעים אינם
 * מכילים הצהרה מפורשת שלילית" - the model hedged and the check believes the
 * hedge - and g-235, whose gold value "מספר פעולות" is arguably not an answer.
 *
 * Derivation, not a run result: 3 rows is ~1.1 points on the 272 denominator of
 * the 10/09 gate run, which was itself invalid.
 *
 * Production path, 14/09/2026: both flags on, the 7 gold + 13 control questions
 * above (--gold/--controls subset files), fresh answers, eval/results/
 * 2026-09-14T18-01-20.json, valid, reranked 7/7, $0.63. The check ran on 18
 * refusals, 0 failed, 0 refusals showed sources without "answers". Controls:
 * refused 13/13; the quote test alone would have kept a source on 12 of them,
 * with the check on 0. Gold: kept g-027 and g-218 (both correct), cleared g-052
 * g-177 g-190 g-235; g-223 did not refuse this time. On the same texts: both
 * flags off 1/6 correct, quote test alone 5/6 (bench-derived, not run), with
 * the check 3/6. So the check trades 2 gold rows for 12 honest controls - the
 * two it drops are g-190 and g-235 again. Still OFF: a gate run decides.
 */
export const REFUSAL_QUOTE_CHECK = env("REFUSAL_QUOTE_CHECK", "0") !== "0";

/**
 * REFUSAL_OFFERS_TOPICS — a refusal the model wrote ANYWAY, after retrieval was
 * confident, hands its candidates back as clickable topics.
 *
 * The case that produced this, reproduced 11/09/2026 on "מה התעריף של מטמיע
 * מערכות?": best cosine 0.390, comfortably over MIN_SCORE, the ranker ran, and
 * the model still opened with "לא מצאתי" — then spent a paragraph explaining
 * that the annexes DO list a tariff for "מדריך / מטמיע", code 4.3. It found the
 * answer and refused to own it. Worse, `sources` came back empty, so the one
 * thing the reader could act on — the document — was not clickable.
 *
 * The clarification machinery for exactly this already exists (clarify.ts,
 * renderClarify), it simply never fires here: `decide` only clarifies in the
 * band BELOW MIN_SCORE, and this question is above it.
 *
 * THE GUARD THAT MATTERS: this fires only when the confidence gate already
 * PASSED. A question the corpus does not cover is refused deterministically
 * before the model is ever called, and never reaches this line — so this cannot
 * become a way to coax an answer out of a corpus that has none, which is the
 * one thing clarify.ts forbids in its own header.
 *
 * Default off: it changes what the UI renders, and iron rule 10 says a new
 * component stays dark until a measured run approves it.
 */
export const REFUSAL_OFFERS_TOPICS = env("REFUSAL_OFFERS_TOPICS", "0") !== "0";

/**
 * JUDGE_WITH_QUESTION — the eval judge sees the QUESTION, not only the value.
 *
 * Not a pipeline component: it changes nothing a user sees. It changes the
 * ruler, which is why it sits behind a flag and is named in describeConfig —
 * a results file must say which judge scored it, or two accuracy numbers get
 * compared across a definition change nobody can see.
 *
 * The defect, found 13/09/2026 at zero cost in the 10/09 gate run: of the six
 * rows filed as "right instruction, wrong value", the answer text of g-154,
 * g-182 and g-213 opens with an explicit "לא" to a yes/no question whose
 * expected value is "לא", and g-196 says "אינו רשאי" against "אינם רשאים" -
 * all four with the expected instruction cited. The model judge rejected all
 * four. It was handed `העובדה: לא` and an answer, with no question: "לא" with
 * no question is not a fact, so the judge had nothing to match it against.
 *
 * Scope, decided by measurement: the question is given ONLY when the fact is
 * exactly "כן" or "לא". Given for every fact, it rejected right answers with a
 * self-readable value (g-291 "4 פעמים בשנה" vs "מדי 3 חודשים"). The numbers
 * are in judge.ts above YES_NO_FACT. Default off; see the memlog of 13/09/2026.
 *
 * Reverse direction checked 15/09/2026 (judge.ts, same place): with the true
 * value it accepts 16 of 18 stored answers, with the flipped value 0 of 18, so
 * it classifies rather than answering the yes/no question itself. Cleared for
 * the next gate run.
 */
export const JUDGE_WITH_QUESTION = env("JUDGE_WITH_QUESTION", "0") !== "0";

/**
 * JUDGE_QUESTION_POLAR_KIND — widen the question-aware judge from facts that
 * are exactly "כן"/"לא" to every gold record whose kind is "כן/לא".
 *
 * Like JUDGE_WITH_QUESTION this changes the ruler, not the pipeline: nothing a
 * user sees moves. Named in describeConfig for the same reason.
 *
 * The defect (queue item 3, 17/09/2026): g-196 answers "לא. בן משפחה ... אינו
 * רשאי לנהוג" against the value "אינם רשאים", cites the expected instruction,
 * and was scored wrong in all four runs that produced that text. Its value is a
 * polar verb phrase - "they are not permitted" says nothing without the question
 * of WHAT is not permitted - which is the exact condition that put the question
 * in front of the judge for "כן"/"לא" on 13/09. The kind field already names
 * that population: 31 records, 6 of them with a bare "כן"/"לא".
 *
 * Measured on stored answers only (eval/benches/2026-09-17-judge-polar-kind):
 * the 9 distinct stored texts this scope newly reaches, each judged with its
 * true value and a flipped one, plus every widened question against every other
 * kind-כן/לא answer of the valid 4-flag gate run. Numbers in judge.ts above
 * YES_NO_FACT. Default off: turning it on is Elad's call, like the others.
 *
 * Standalone: on by itself it covers the bare "כן"/"לא" facts too (all six are
 * kind "כן/לא"), so describeConfig reports the wider scope whichever of the two
 * flags is also set.
 */
export const JUDGE_QUESTION_POLAR_KIND = env("JUDGE_QUESTION_POLAR_KIND", "0") !== "0";

/**
 * UNVERIFIED_ALLOWS_PASSAGE_REFS — a citation of an instruction that a passage
 * shown to the model itself names is not reported as unverified.
 *
 * The defect, found 15/09/2026 at zero cost in the valid gate run
 * (eval/results/2026-09-15T10-52-52.json): g-260 answers "20 בנובמבר" from the
 * table in instruction 1.6.3, cites 1.6.3, and was scored wrong only because
 * hasUnverified=true. The flagged number was 3.2.12 - the model wrote "הטבלה
 * מציינת שהוראה 12.3.1 וכן הוראה 3.2.12 הן ההוראות הרלוונטיות", which is what
 * the table in 1.6.3 says. 3.2.12 was not retrieved, so unverifiedCitations
 * reported it as an invented citation, and the reader saw a warning on a
 * correct, correctly sourced answer.
 *
 * unverifiedCitations already documents the intent this restores: "a
 * cross-reference the passage itself makes" should not be flagged (rule 6 of
 * the prompt asks for it). The implementation only ever checked the RETRIEVED
 * codes, never the passage text, so the intent was never true.
 *
 * What the warning still catches with the flag on: a number that appears in no
 * passage at all - the 7.3.7 case it was written for (answer.ts, above the
 * allowed-list rule). What it stops catching: a fact attached to an instruction
 * that a passage merely mentions. That is the trade, and why it is a flag.
 *
 * Derived on stored answers before any code: +1 gold row (g-260) on the gate
 * run. Default off (iron rule 10).
 */
export const UNVERIFIED_ALLOWS_PASSAGE_REFS = env("UNVERIFIED_ALLOWS_PASSAGE_REFS", "0") !== "0";

/**
 * SOURCES_ADD_QUOTED_DOC — an answer that QUOTES a passage gets that passage's
 * document in its sources, whatever number the model wrote next to the quote.
 *
 * The defect, sorted 16/09/2026 (eval/benches/2026-09-16-cited-other/): g-096
 * copies "במקרה שהשליח האריך את שליחותו, לא יהא זכאי לחופשה נוספת" word for
 * word - a sentence that exists in exactly one instruction, 14.2.10 - and labels
 * it "הוראה 14.2.8". citedHits reads only the label, so the sources box, the
 * part a clerk checks against the binding original, never names 14.2.10.
 * 14.2.10#7 is a flattened table row that prints "מס' 14.2.8" inside itself.
 *
 * g-278 has the same mechanism once its content is right. Probe 16/09/2026
 * (eval/benches/2026-09-16-reversed-ranges/range-probe.mts, 3 calls per arm,
 * identical outputs): with 12.6.4#26's reversed range written in order, the
 * model quotes 12.6.4 verbatim and labels the quote "הוראה 7.7.1" three times
 * of three. The 10/09 gate row did the same with the range as it is.
 *
 * ADDITIVE ONLY, and that is the measured part. A rule that REMOVES a code the
 * model cited because the code appears only as a cross-reference would touch 18
 * of the 259 correct gate answers (6.9%). Adding the quoted document changes no
 * text, no warning and no refusal; it only puts the true origin beside the label.
 * What it does NOT fix: the text still says "הוראה 14.2.8". The reader now has
 * the right document to click, not a corrected sentence.
 *
 * Derived on stored answers before any code (eval/benches/2026-09-16-quote-
 * attribution/derive.mts): gate 2026-09-15T14-22-16 259 -> 260 (g-096), zero
 * rows down, zero answered controls changed; 10/09 gate +g-278. It needed the
 * gershayim fix in quotedSpans (answer.ts) to see g-096 at all.
 *
 * Refusals are excluded - they have their own path (REFUSAL_KEEPS_QUOTED_SOURCES).
 * Evidence field: Answer.quotedDocsAdded. Default off (iron rule 10).
 */
export const SOURCES_ADD_QUOTED_DOC = env("SOURCES_ADD_QUOTED_DOC", "0") !== "0";


// ---------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------

/**
 * Chunk size is capped by the embedding model's context window, not by taste:
 * Cohere Multilingual v3 accepts ~512 tokens, and Hebrew runs roughly 2-3
 * characters per token, so 1200 characters could silently overflow and be
 * truncated. 900 keeps a full clause together while staying inside the window.
 * Story 1.2 measures the real ratio and this number moves if the data says so.
 */
/**
 * Cut chunks on the document's own clause boundaries rather than on a character
 * count. See clauseBlocks() in chunk.ts for the reasoning and the measurement.
 *
 * A flag rather than a rewrite because it changes what every vector in the
 * index represents: switching it requires a full re-ingest, and being able to
 * put the old behaviour back in one environment variable is what makes the
 * before/after comparison honest.
 */
/**
 * A clause block shorter than this is a stub, not a passage, and is merged into
 * the block after it. See clauseBlocks() in chunk.ts.
 *
 * 150 comes from the measured damage: at 0 (no merging) 37% of the index came
 * out under 100 characters and the median was 141, because a heading such as
 * "מבוא" or "הגדרות" is a depth-1 clause exactly like a rule is. Those
 * six-character vectors match every question weakly and none of them well.
 */
export const MIN_BLOCK_CHARS = envNum("MIN_BLOCK_CHARS", 150);

export const STRUCTURAL_CHUNKING = env("STRUCTURAL_CHUNKING", "1") !== "0";

export const CHUNK_CHARS = envNum("CHUNK_CHARS", 900);
export const CHUNK_OVERLAP = envNum("CHUNK_OVERLAP", 180);

/** How many passages are handed to the model per question. */
export const TOP_K = envNum("TOP_K", 5);

/**
 * At most this many passages from any single document.
 *
 * Measured on the real corpus, not assumed: without a cap, all five slots for
 * "when is a tender exemption allowed and who approves it?" came back from
 * instruction 7.6.1 — five near-identical passages saying the same thing, while
 * 7.1.1 (which is what actually names the approving committee) never made the
 * list. A long document simply has more chances to rank.
 *
 * Capping per document costs a little depth on single-source questions and buys
 * the cross-referencing that makes this worth building: the answer to a real
 * procurement question usually lives in two instructions, sometimes in two
 * different chapters.
 */
export const MAX_PER_DOC = envNum("MAX_PER_DOC", 2);

/**
 * Below this cosine score the passages are treated as unrelated and the system
 * refuses instead of inviting the model to improvise (KR2).
 *
 * RE-CALIBRATED 30/08/2026 for Titan v2. A threshold belongs to the embedding
 * model that produced the index, not to the corpus — carrying 0.60 over from
 * Cohere would have refused every demo question except the first.
 *
 * Measured on the rebuilt index, best score per question:
 *   covered    0.756  0.684  0.616  0.570   ← the four demo questions
 *   uncovered  0.411  0.337  0.287  0.165   ← leave days, passports, tax, aircon
 * The usable gap is 0.411–0.570, and 0.49 is its midpoint.
 *
 * Under Cohere the same two bands were 0.595–0.822 and up to 0.545: a gap of
 * 0.05, with an unrelated instruction landing one hundredth below the gate. The
 * gap is now 0.159 — three times wider — which is the whole reason the embedding
 * model changed. The refusal guarantee stopped being a coin flip.
 *
 * Note the direction of the change: the number went DOWN while the system became
 * STRICTER. An absolute threshold is meaningless across models; only its position
 * between the two measured bands means anything.
 *
 * The threshold is deliberately the cheap first line of defence, not the only
 * one: the system prompt still forbids answering from anything but the passages.
 */
/**
 * How many chunks on each side of a hit are handed to the model as context.
 *
 * 1 means a hit arrives as a trio: the passage before it, the passage itself,
 * and the passage after. See withNeighbours() in retrieve.ts for why — the short
 * version is that similarity finds the heading and the answer is often the
 * sentence just past it, and a 900-character window with a 2-per-document cap
 * was showing the model ~1,800 characters of an instruction 20x that long.
 *
 * Set to 0 to restore the old behaviour exactly. Raising it past 2 stops paying:
 * the prompt grows linearly while the chance the answer is that far from any
 * matching sentence falls off fast.
 */
/**
 * Hand the model the WHOLE instruction a hit came from, not just its
 * neighbouring chunks. See withParentDocuments() in retrieve.ts.
 *
 * On by default: a regulation scatters the rule, its conditions, its exceptions
 * and the approving authority across adjacent clauses, and a ±1 window can stop
 * one clause short of the one that changes the answer.
 */
/**
 * Run a BM25 lexical engine alongside the vectors and fuse the two rankings.
 *
 * See hybridFor() in retrieve.ts. Safe only because the refusal decision now
 * belongs to the model rather than to a similarity score — this is the same
 * feature that had to be reverted when a score threshold was the gate.
 */
/**
 * Rewrite the question into the regulator's vocabulary before retrieval and
 * search both phrasings. See expandVocabulary() in query-prep.ts.
 *
 * One extra Haiku call per question (~$0.0006, ~600ms). Safe only because the
 * refusal gate is the model: a rewrite of an out-of-scope question produces
 * fluent bureaucratic Hebrew that WILL retrieve something, and a threshold
 * would have answered it. Re-measure the 50 controls after any prompt change.
 *
 * OFF by default since 07/09/2026, measured on 76 citizen-phrased questions
 * over the whole-corpus search: top-1 41 -> 44 but top-3 62 -> 57. The model
 * reads the top documents, not only the first, so losing five top-3 hits to
 * gain three top-1 hits is a worse answer more often than a better one. The
 * losses were rewrites that invented context the question never had. Add to
 * that 600ms and a Haiku throttle that silently falls back to the original
 * question, and the bridge for the vocabulary gap belongs AFTER retrieval (a
 * reranker over the candidates), not before it. The code stays for A/B runs:
 * QUERY_EXPANSION=1.
 */
export const QUERY_EXPANSION = env("QUERY_EXPANSION", "0") !== "0";

export const HYBRID_SEARCH = env("HYBRID_SEARCH", "1") !== "0";

/** How many lexical candidates per collection enter the fusion. */
export const HYBRID_CANDIDATES = envNum("HYBRID_CANDIDATES", 10);

/**
 * Stage 1½ — rerank the candidates with the generation model (rerank.ts).
 *
 * Why, in one number pair from 07/09/2026 (76 citizen-phrased questions, whole-
 * corpus search): the right instruction is in the top three 82% of the time
 * but first only 54% of the time. Search finds it and then loses the coin toss
 * against its neighbours. A reranker reads question and passage together and
 * decides "does this contain the answer", which neither cosine nor BM25 asks.
 *
 * RERANK_CANDIDATES is how wide a net the search casts for the reranker to
 * sort; the answer still sees TOP_K passages afterwards, so cost and context
 * size at the generation stage do not change. The reranker itself is one short
 * Haiku call per question (measured: see the memlog entry of the same day).
 */
/**
 * Whether the generated per-chunk context sentence joins the embedded text.
 *
 * Default OFF. The sentences can be generated and stored without changing a
 * single search result; flipping this is what makes them count, and it only
 * flips once the measurement says it should. Turning it on requires a full
 * re-embed — the vectors change — so this is not a runtime toggle in the way
 * the others are.
 */
export const CONTEXTUAL_RETRIEVAL = env("CONTEXTUAL_RETRIEVAL", "0") !== "0";

export const RERANK = env("RERANK", "1") !== "0";
export const RERANK_CANDIDATES = envNum("RERANK_CANDIDATES", 12);

/**
 * Parent vote: does agreement between sibling chunks count as a ranking signal?
 *
 * A TAKAM instruction is split into many chunks (955 documents, 28,514 chunks,
 * median 12 chunks per document). When a question is really about instruction
 * X, it is common for two or three of X's chunks to land in the candidate list
 * without any single one of them scoring highest. That agreement is evidence,
 * and today it is thrown away entirely: withParentDocuments runs AFTER the
 * ranking has already chosen, so it can widen the winner but never change who
 * won.
 *
 * What this does: before the reranker sees the list, chunks are grouped by the
 * document they came from. A document that qualifies is collapsed to ONE entry
 * carrying its best chunk, and the qualifying documents are moved to the front
 * as a block. It is a PRIOR, not a verdict — the reranker still reads the text
 * and may put something else first.
 *
 * Measured target: on the 08/09 baseline, 34 of 71 failures had the correct
 * instruction at rank 2 and another 13 at rank 3. That is a first-versus-second
 * problem, which is exactly what sibling agreement speaks to.
 *
 * Source: LlamaIndex AutoMergingRetriever (simple_ratio_thresh=0.5).
 * Default OFF until a measurement says otherwise (rule 10).
 */
export const PARENT_VOTE = env("PARENT_VOTE", "0") !== "0";

/**
 * How many of a document's chunks must be in the candidate list before the
 * document is promoted, as a fraction of all the chunks it has.
 *
 * MEASURED CAVEAT, and the reason PARENT_VOTE_MIN_TOTAL exists below: with a
 * median of 12 chunks per document and a 12-candidate list, this ratio can
 * essentially never fire for a normal document — it would take 7 of one
 * instruction's 12 chunks to fill more than half the list. Where it DOES fire
 * unconditionally is the 113 single-chunk documents (11.8% of the corpus),
 * whose single chunk is always 100% of them. Promoting a document because it
 * is short is not agreement; it is a constant. So the ratio keeps its meaning
 * for documents that have siblings, and documents without siblings are judged
 * on their score like before.
 */
export const PARENT_VOTE_RATIO = Number(process.env.PARENT_VOTE_RATIO ?? 0.5);

/** Chunks from one document in the list that qualify it outright. */
export const PARENT_VOTE_MIN_CHUNKS = envNum("PARENT_VOTE_MIN_CHUNKS", 2);

/** A document with fewer chunks than this has no siblings to agree with it. */
export const PARENT_VOTE_MIN_TOTAL = envNum("PARENT_VOTE_MIN_TOTAL", 2);

/**
 * How strong the agreement bonus is, as the K of a reciprocal-rank sum: each
 * chunk of a document contributes 1/(K + its position), and the document is
 * ordered by the total.
 *
 * Small K = a nudge that can only lift a document already near the top; large K
 * flattens the positions until any two chunks beat any one chunk, which is the
 * "promote to the front as a block" behaviour. This was measured, not chosen:
 * block promotion moved the correct instruction DOWN three times for every once
 * it moved it up (20 gold questions, retrieval only, no cost), because a pair of
 * chunks at positions 7 and 9 was overtaking a single chunk at position 1.
 */
export const PARENT_VOTE_K = envNum("PARENT_VOTE_K", 3);

export const PARENT_CONTEXT = env("PARENT_CONTEXT", "1") !== "0";

/**
 * Character budget for that expansion, across all documents in one answer.
 *
 * 24,000 characters is roughly 9,000 Hebrew tokens — close to what a question
 * already costs today, so this changes WHICH text the model reads far more than
 * how much. Instructions run to 30,000 characters on their own, so without a
 * cap a single long one would consume the whole request.
 */
export const PARENT_CONTEXT_CHARS = envNum("PARENT_CONTEXT_CHARS", 24000);

/**
 * PARENT_TOP_DOC_SLACK — the top-ranked document goes in WHOLE when it is over
 * the budget by at most this fraction (0.25 = up to 30,000 of 24,000). 0 = off.
 *
 * The defect, found 15/09/2026 at zero cost on the valid 4-flag gate run
 * (eval/results/2026-09-15T14-22-16.json): g-021 asks how many times last
 * year's activity cost may be approved, the answer ("פי שניים", clause 1.3.2)
 * is in 6.1.1, 6.1.1 ranked first — and the model wrote that the passages do
 * not contain it. They did not. 6.1.1 is 26,153 characters against 24,000, so
 * it went in as an excerpt: its head (chunks 0-18) plus a window around the two
 * chunks the search matched (41-42, the payment rules). The fact sits in chunk
 * 33, in the gap. The chunk is a near-verbatim match yet ranks 11th inside its
 * own document by cosine, so the search could not have pulled it in either.
 * Bench: eval/benches/2026-09-15-g021-context/.
 *
 * Why slack on the first document and not a bigger budget. Scanned on all 300
 * gold rows with the production withParentDocuments, no model call: at 24,000
 * g-021 is the ONLY checkable row (of 260) whose fact is missing from the
 * passages. At 28,000 it is fixed and g-212 — correct today — loses its fact,
 * because a larger budget lets an earlier document in whole and starves the
 * one below it. At 32,000 no row misses, at +24% characters on every question.
 * The slack pays only where the top document barely overflows.
 *
 * The trade: when the top document is the WRONG one, including it whole leaves
 * the documents below it only their matched chunks. Default off (iron rule 10).
 */
export const PARENT_TOP_DOC_SLACK = Math.max(0, envNum("PARENT_TOP_DOC_SLACK", 0));

export const NEIGHBOUR_RADIUS = envNum("NEIGHBOUR_RADIUS", 1);

/**
 * The lexical table assist: may a phrase match pull a table row into the
 * passages shown to the model?
 *
 * On by default, because the problem it solves is real — a bare rate table has
 * no semantic surface and cannot be reached any other way. It is a flag rather
 * than a constant because it has already been reverted once and reintroduced
 * once, and the next person to suspect it should be able to turn it off in one
 * environment variable instead of editing retrieval code.
 *
 * It can no longer affect the refusal decision; see isConfident in retrieve.ts.
 */
export const LEXICAL_TABLE_ASSIST = env("LEXICAL_TABLE_ASSIST", "1") !== "0";

/** How many table rows the assist may append beyond the ranked passages. */
export const LEXICAL_TABLE_MAX = envNum("LEXICAL_TABLE_MAX", 3);

/**
 * The clarification layer — see clarify.ts for the measurements behind these.
 *
 * CLARIFY_MIN_SCORE is the floor below which we refuse rather than offer
 * topics: 0.40, because under it only 15% of real questions had an answer in
 * the top document and 46% anywhere in the top three, so a topic list there is
 * mostly noise. Between 0.40 and the answer threshold, the top three held the
 * answer 71% of the time — that band is the whole point of the layer.
 */
/**
 * Use remembered clarification choices to ORDER the topics offered.
 *
 * On by default because the risk is contained: it changes which topic appears
 * first, and the person still decides. It never reaches the refusal gate.
 */
export const FEEDBACK_ORDERING = env("FEEDBACK_ORDERING", "1") !== "0";

/**
 * How similar a new question must be to a remembered one to reuse its choice.
 *
 * 0.92 is deliberately strict — near-paraphrase, not "same topic". A looser bar
 * turns one person's answer into a standing rule for a family of questions,
 * which is a wrong answer nobody can trace back to a decision.
 */
export const FEEDBACK_SIMILARITY = Number(process.env.FEEDBACK_SIMILARITY ?? 0.92);

export const CLARIFY_MIN_SCORE = Number(process.env.CLARIFY_MIN_SCORE ?? 0.95);

/** How many topics to offer. Beyond four a list stops being a choice. */
export const CLARIFY_MAX_TOPICS = envNum("CLARIFY_MAX_TOPICS", 3);

/**
 * Offer topics even when the score alone would justify answering.
 *
 * On by default, and the reason is a measurement rather than caution: above
 * 0.55 the top document still only answered 55% of questions. A high score
 * means "this text resembles the question", not "this is the document you
 * wanted", and when several distinct topics are in play the person asking is
 * the only one who can tell them apart.
 */
export const CLARIFY_ALWAYS = env("CLARIFY_ALWAYS", "0") !== "0";

/**
 * The cosine floor below which the model is never called.
 *
 * Lowered from 0.49 to 0.30 on 06/09/2026, and the reason is a measurement that
 * reverses the assumption the number was built on.
 *
 * 0.49 was calibrated to separate covered questions from uncovered ones. On a
 * 126-question benchmark written in real user wording it does not: covered
 * questions run 0.32–0.75 and adjacent out-of-scope ones reach 0.62, so the
 * bands overlap and no threshold splits them. What the threshold DID do
 * reliably was refuse things it should have answered — the median real question
 * scored 0.489, right on the line — and it was structurally incapable of
 * passing a rate table, whose rows ("בודק תוכנה  170  253") have no sentence
 * for an embedding to resemble and score around 0.32 however relevant they are.
 *
 * Measured alternative: open the gate and let the MODEL decide. Across all 50
 * deliberately out-of-scope control questions it refused 50 and answered none,
 * while correctly answering the rate lookup the threshold had been blocking. A
 * model that reads the passage beats a number that only measures resemblance.
 *
 * So this is no longer the relevance decision. It is a floor whose only job is
 * to stop a model call when retrieval returned nothing remotely related — the
 * lowest control question scored 0.249, so 0.30 keeps that free while letting
 * everything the model could plausibly judge through. The refusal contract in
 * the prompt is now what enforces KR2.
 *
 * The cost of the change is real and worth stating: a refusal used to be free
 * and now costs about $0.012, because the model is actually called.
 */
export const MIN_SCORE = Number(process.env.MIN_SCORE ?? 0.30);

/**
 * The same threshold measured again on the English AWS collections, because the
 * Hebrew calibration above does NOT transfer — and the reason is in the comment
 * itself: 0.6 was chosen to clear a Hebrew *floor* of ~0.50 that exists because
 * any two Hebrew administrative texts already look alike to a multilingual
 * model. English technical documentation has no such floor, so both bands sit
 * lower and further apart.
 *
 * Measured 23/08/2026 against aws-lambda, 5 covered questions and 4 deliberately
 * uncovered ones:
 *   covered    0.595 – 0.822   (lowest: "cold start", which the docs call
 *                               "execution environment lifecycle" — a real
 *                               vocabulary mismatch, and exactly the case a
 *                               threshold must not throw away)
 *   uncovered  0.271 – 0.395   (Holocaust-survivor stipend, sourdough bread,
 *                               a Hebrew TAKAM question, the offside rule)
 *
 * The gap is 0.20 wide, against 0.105 in Hebrew. 0.5 is its midpoint: 0.105
 * above the highest uncovered score and 0.095 below the lowest covered one.
 * Keeping 0.6 here refused a legitimate question while still admitting nothing
 * extra — strictness bought no safety and cost real answers.
 *
 * RE-MEASURED 30/08/2026 on aws-kendra after its rebuild on Titan v2:
 *   covered    0.698 – 0.784
 *   uncovered  0.076 – 0.140   (sourdough, capital of France, changing a tyre)
 * A 0.558 gap — nearly three times the Cohere one, and the same story the Hebrew
 * calibration told. 0.5 stays: it sits inside BOTH gaps, which matters while the
 * English collections migrate one at a time and the two models coexist.
 *
 * Note the asymmetry worth remembering: Hebrew needed the threshold MOVED when
 * the model changed (0.60 → 0.49), English did not. Cohere's weakness was
 * specific to Hebrew administrative text, where shared boilerplate pulled any
 * two documents together. That is why the model was measured per language rather
 * than judged by one headline number.
 */
export const MIN_SCORE_EN = Number(process.env.MIN_SCORE_EN ?? 0.5);

/**
 * One threshold per language, not one per corpus: the number is a property of
 * how the embedding model spaces that language, so a second Hebrew corpus would
 * inherit the Hebrew number for free.
 */
export const minScoreFor = (collection: string): number =>
  collectionDef(collection).language === "he" ? MIN_SCORE : MIN_SCORE_EN;

// ---------------------------------------------------------------------------
// Secrets — read here, never logged, never written to an artefact
// ---------------------------------------------------------------------------

export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
export const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY ?? "";
export const GEMINI_API_KEY = process.env.GEMINI_API_KEY ?? "";

/** One-line banner so every run states what it is actually talking to. */
export function describeConfig(): string {
  return [
    `mode=${RUN_MODE}`,
    `embed=${EMBED_TRANSPORT} (${EMBED_MODELS[EMBED_TRANSPORT].modelId} @ ${EMBED_REGION})`,
    `llm=${LLM_ROUTE} (${GEN_MODELS[LLM_ROUTE].modelId}), genRegions:${GEN_REGIONS.length ? GEN_REGIONS.join("+") : "off"}`,
    `bucket=${S3_BUCKET} @ ${S3_REGION}`,
    /*
     * Every pipeline flag is named here, ON OR OFF.
     *
     * The previous version of this block said "a results file whose settings
     * are unknown is not a measurement" and then listed a flag only while it
     * was ON, which for a reader of an old file is the same as not listing it.
     * Measured 10/09/2026: recall@12 moved 96.3% -> 98.3% between the 08/09
     * baseline and the 10/09 gate run, and neither results file records a
     * single retrieval setting, so nothing on disk can separate a real gain
     * from a knob that moved. A default is a setting too, and defaults change.
     */
    `retrieve=topK:${TOP_K},minScore:${MIN_SCORE}/${MIN_SCORE_EN},maxPerDoc:${MAX_PER_DOC},hybrid:${HYBRID_SEARCH ? HYBRID_CANDIDATES : "off"},expand:${QUERY_EXPANSION ? "on" : "off"}`,
    `rank=rerank:${RERANK ? RERANK_CANDIDATES : "off"},parentVote:${PARENT_VOTE ? `on(min=${PARENT_VOTE_MIN_CHUNKS},ratio=${PARENT_VOTE_RATIO},k=${PARENT_VOTE_K})` : "off"}`,
    `answer=parentDocs:${PARENT_CONTEXT ? PARENT_CONTEXT_CHARS : "off"},neighbours:${NEIGHBOUR_RADIUS},style:${ANSWER_STYLE},applicability:${ANSWER_APPLICABILITY ? "on" : "off"},refusalQuotedSources:${REFUSAL_KEEPS_QUOTED_SOURCES ? "on" : "off"},refusalQuoteCheck:${REFUSAL_QUOTE_CHECK ? "on" : "off"},refusalTopics:${REFUSAL_OFFERS_TOPICS ? "on" : "off"},passageRefs:${UNVERIFIED_ALLOWS_PASSAGE_REFS ? "on" : "off"},topDocSlack:${PARENT_TOP_DOC_SLACK || "off"},quotedDocSources:${SOURCES_ADD_QUOTED_DOC ? "on" : "off"},scopeFirst:${ANSWER_SCOPE_FIRST ? "on" : "off"}`,
    `judge=question:${JUDGE_QUESTION_POLAR_KIND ? "yes-no-kind" : JUDGE_WITH_QUESTION ? "yes-no-facts" : "off"}`,
  ].join("  |  ");
}
