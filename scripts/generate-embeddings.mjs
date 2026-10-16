import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pipeline } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/all-MiniLM-L6-v2-ONNX";
const MODEL_DTYPE = "q4";
const BATCH_SIZE = 12;
const DATA_DIR = path.resolve(
  process.cwd(),
  "public",
  "data",
  "Aequitas_Research_Intelligence_Assignment_Data",
);
const FEED_PATH = path.join(DATA_DIR, "research_feed.csv");
const OUTPUT_PATH = path.join(DATA_DIR, "research_embeddings.json");

function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  const input = text.replace(/^\uFEFF/, "");

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    const next = input[index + 1];

    if (char === '"') {
      if (quoted && next === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell.length || row.length) {
    row.push(cell);
    if (row.some((value) => value.trim())) rows.push(row);
  }

  if (!rows.length) return [];
  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = (values[index] ?? "").trim();
    });
    return record;
  });
}

function companyOf(record) {
  if (record.company?.trim()) return record.company.trim();
  if (record.headline?.includes(":")) return record.headline.split(":")[0].trim();
  return "";
}

function embeddingText(record) {
  const fields = [
    `Headline: ${record.headline ?? ""}`,
    `Company: ${companyOf(record)} ${record.ticker ?? ""}`.trim(),
    `Sector: ${record.sector ?? ""}; Industry: ${record.industry ?? ""}`,
    `Tags: ${record.tags ?? ""}`,
    `Summary: ${(record.content ?? "").slice(0, 1100)}`,
  ];
  // Keep the most useful information at the start because encoder inputs have a token limit.
  return fields.filter((field) => !field.endsWith(": ")).join(". ").slice(0, 1800);
}

async function main() {
  console.log(`Reading research feed: ${FEED_PATH}`);
  const csv = await readFile(FEED_PATH, "utf8");
  const records = parseCSV(csv).filter((record) => record.record_id && record.headline);
  if (!records.length) {
    throw new Error("No research records found. Check the dataset folder and research_feed.csv.");
  }

  console.log(`Found ${records.length} records.`);
  console.log(`Loading ${MODEL_ID} (${MODEL_DTYPE}) to generate local embeddings…`);
  const extractor = await pipeline("feature-extraction", MODEL_ID, { dtype: MODEL_DTYPE });
  const embeddings = {};

  for (let start = 0; start < records.length; start += BATCH_SIZE) {
    const batch = records.slice(start, start + BATCH_SIZE);
    const output = await extractor(batch.map(embeddingText), { pooling: "mean", normalize: true });
    const vectors = output.tolist();

    if (!Array.isArray(vectors) || vectors.length !== batch.length) {
      throw new Error(`Unexpected embedding output for batch starting at record ${start + 1}.`);
    }

    batch.forEach((record, index) => {
      const vector = vectors[index];
      if (!Array.isArray(vector) || vector.length !== 384) {
        throw new Error(`Invalid embedding vector for ${record.record_id}. Expected 384 dimensions.`);
      }
      embeddings[record.record_id] = vector.map((value) => Number(value.toFixed(6)));
    });

    console.log(`Embedded ${Math.min(start + batch.length, records.length)}/${records.length} records`);
  }

  const indexFile = {
    model: MODEL_ID,
    modelDtype: MODEL_DTYPE,
    dimensions: 384,
    recordCount: Object.keys(embeddings).length,
    generatedAt: new Date().toISOString(),
    embeddings,
  };

  await writeFile(OUTPUT_PATH, `${JSON.stringify(indexFile)}\n`, "utf8");
  console.log(`\nDone. Wrote ${Object.keys(embeddings).length} vectors to:`);
  console.log(OUTPUT_PATH);
  console.log("Keep this JSON file in public/data and commit it so Vercel can serve it.");
}

main().catch((error) => {
  console.error("\nEmbedding generation failed:");
  console.error(error instanceof Error ? error.message : error);
  console.error("Make sure @huggingface/transformers is installed and the dataset path is correct.");
  process.exitCode = 1;
});
