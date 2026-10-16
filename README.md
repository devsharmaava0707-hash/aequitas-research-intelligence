# Aequitas Research Intelligence

A research-intelligence dashboard for browsing, prioritising, clustering, and querying a supplied company-research feed. It combines conventional filters and transparent priority heuristics with local semantic retrieval and a citation-based Q&A assistant.

**Live app:** https://aequitas-research-intelligence.vercel.app  
**Repository:** https://github.com/devsharmaava0707-hash/aequitas-research-intelligence

> **Data note:** This project uses fictionalised assignment data. Priority scores are navigation aids, not investment advice. AI answers are intended to be grounded in retrieved records and should be checked against their cited sources.

---

## Features

### 1. Research feed and filters
- Browse the research records supplied with the assignment.
- Filter by publication date, category/source type, sector, company, priority, and tags.
- Sort by newest, oldest, or priority.
- Search headlines, content, companies, tickers, sectors, industries, sources, countries, and tags.
- Highlight matching search terms in results.
- Open any record in the reading pane, inspect its metadata, and open its bundled source document.

### 2. Needs attention and priority triage
- Scores each record using a transparent, rule-based keyword heuristic covering financial results, regulatory/legal developments, corporate actions, credit signals, and source types.
- Shows a compact **Needs attention** section with up to three distinct companies.
- Displays priority indicators in the feed to help surface records for review.
- Shows the reason and heuristic score for a selected record.

The score is not a predictive model, investment recommendation, or judgement of a company's value.

### 3. Near-duplicate clustering
- Groups likely related announcements in the main research feed so repeated updates do not occupy as many full rows.
- Uses multiple signals including company/ticker identity, headline and content token similarity, tag overlap, event-family rules, publication dates, and geography safeguards.
- Keeps the strongest representative visible and provides a **`+N related`** control to expand the underlying records.
- Uses event categories to avoid casually merging different story types, such as network rollout, spectrum auctions, ARPU/tariff updates, and financial results.
- Related records remain individually accessible and can be opened or saved separately.

Clustering is a heuristic: review the expanded records before treating a cluster as a single event.

### 4. Semantic + keyword search
- Uses a local embedding model, `onnx-community/all-MiniLM-L6-v2-ONNX`, to represent the supplied records and search query as vectors.
- Blends semantic similarity with lexical/keyword relevance so searches can find conceptually related records even when wording differs.
- Precomputes record embeddings into `research_embeddings.json` so the whole dataset does not need to be embedded every time the page loads.
- Runs query embedding and ranking in the browser after the model is available; the model is downloaded on first use and may take a little while to initialise.
- Falls back to keyword search if the semantic index or local model is unavailable.

Semantic search requires both the embedding-generation script and the generated index described below. The embeddings JSON is data, not a credential, and must be present in `public/data` in deployed builds.

### 5. Ask the Feed — AI Q&A with citations
- Lets users ask natural-language questions about the research feed.
- Retrieves up to 12 relevant records, preferring semantic retrieval when available and falling back to keyword retrieval.
- Sends only selected record IDs to the server route. The route reloads the canonical records from the deployed CSV rather than trusting client-submitted record text.
- Calls Cloudflare Workers AI using a hosted Meta Llama model.
- Instructs the model to answer only from supplied records, use exact record IDs in citations, acknowledge missing evidence, and avoid investment recommendations.
- Validates returned citation IDs against the records actually supplied and renders citations as links to the corresponding record.
- Keeps Cloudflare credentials on the server in environment variables; no API key is embedded in client-side code.
- Leaves the research feed usable if the AI provider is unavailable.

The current route limits question length, retrieved-source count, context length, and generated answer length. It also has a best-effort in-memory per-client rate limit for a small demo. That rate limiter is not a global abuse-prevention system; add a shared store and stronger abuse protection before high-traffic public use.

### 6. Company profiles and timelines
- Browse the supplied company reference universe.
- View available ticker, sector, industry, geography, description, and alias details.
- Open research records associated with a company.
- Add or remove tracked companies from the watchlist.

### 7. Read later, watchlist, and saved searches
- Bookmark individual research records for later review.
- Maintain a company watchlist.
- Save common searches for quick reuse.
- Persist these preferences in browser `localStorage`.

### 8. Keyboard shortcuts
- `/` — focus search.
- `j` / `k` — move through results.
- `Enter` — open the focused/selected result.
- `b` — bookmark the selected result.
- `Esc` — close Ask the Feed, blur search, or close the current detail pane as applicable.
- `?` — open the shortcuts popover.

---

## Technology

- **Next.js App Router** and **React**
- **TypeScript**
- **Tailwind CSS**
- **Hugging Face Transformers.js** for browser-side embeddings
- **ONNX** model: `onnx-community/all-MiniLM-L6-v2-ONNX`
- **Cloudflare Workers AI** for Ask the Feed generation
- **Meta Llama 3.1 8B Instruct FP8** (configurable through `CLOUDFLARE_MODEL`)
- Static CSV/text files served from `public/data`

No OpenAI SDK or OpenAI API key is required by the Cloudflare version of Ask the Feed.

---

## Getting started

### Requirements

- Node.js version compatible with the installed Next.js version
- npm
- The supplied Aequitas assignment dataset
- Internet access for the first download of the local embedding model and for Cloudflare Workers AI Q&A

### 1. Clone and install

```bash
git clone https://github.com/devsharmaava0707-hash/aequitas-research-intelligence.git
cd aequitas-research-intelligence
npm install
```

### 2. Add the dataset

Keep the dataset under this exact path:

```text
public/data/Aequitas_Research_Intelligence_Assignment_Data/
```

The app expects at least these files:

```text
research_feed.csv
company_master.csv
watchlist.csv
raw_documents/
  RF0001.txt
  ...
```

The record CSV should include the record IDs and source paths referenced by the app. Preserve the source-data filenames and relative paths so record details and original documents can load correctly.

### 3. Set up semantic search embeddings

Install the Transformers.js package if it is not already in `package.json`:

```bash
npm install @huggingface/transformers
```

Ensure this script exists:

```text
scripts/generate-embeddings.mjs
```

Generate the index from the project root:

```bash
node scripts/generate-embeddings.mjs
```

The first run downloads the embedding model and processes the records. On success, it writes:

```text
public/data/Aequitas_Research_Intelligence_Assignment_Data/research_embeddings.json
```

Keep this generated JSON file in the project and commit it so the deployed app can load it. Regenerate the index whenever the underlying research feed changes materially.

### 4. Configure Cloudflare Workers AI (Ask the Feed)

Create a Workers AI API token in Cloudflare and copy the account ID. Then create a **`.env.local`** file in the project root, next to `package.json`:

```env
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_workers_ai_api_token
CLOUDFLARE_MODEL=@cf/meta/llama-3.1-8b-instruct-fp8
```

Replace the first two placeholder values with your actual credentials. Do not add a `NEXT_PUBLIC_` prefix. The variables are consumed by the server API route.

Make sure `.gitignore` contains:

```gitignore
.env.local
.env.*.local
```

Verify the file is ignored:

```powershell
git check-ignore -v .env.local
```

Never commit `.env.local`, paste credentials into the source code, or share API tokens in screenshots or chat. The provided `.env.example` should contain placeholders only.

### 5. Run locally

After saving environment variables, restart the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

To check the production build:

```bash
npm run build
```

### 6. Configure Vercel

For production Ask the Feed requests, add the following environment variables in **Vercel → Project → Settings → Environment Variables**:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_MODEL`

Set them for the environments you use, such as Production and Preview, then redeploy. `.env.local` stays on your computer and is not uploaded by Git. Ensure the CSV dataset, raw source documents, and `research_embeddings.json` are included in the Git repository so Vercel can serve them.

---

## Suggested demo walkthrough

Use this sequence to demonstrate the project's capabilities:

1. **Search and filter:** Search for `OrbitTel`, then filter by category or priority.
2. **Semantic search:** Try `companies expanding network capacity` and compare it with the exact query `orbitel 5G`.
3. **Clustering:** Find the OrbitTel 5G site expansion stories. Expand `+N related` to inspect individual records, and compare them with spectrum auction or ARPU stories to see that distinct event types should remain separate.
4. **Priority triage:** Select a high-priority record and explain the transparent heuristic score and the “Why this was flagged” explanation.
5. **Company research:** Open a company profile, scan its timeline, and add it to the watchlist.
6. **Read later:** Save a story, then open the Read later tab.
7. **Ask the Feed:** Ask `Summarize OrbitTel's network expansion`. Open one of the cited record IDs and verify the underlying research.
8. **Evidence boundary:** Ask for a figure not present in the feed (for example, the total investment amount for an event) and check that the assistant acknowledges when the retrieved records do not establish the answer.

---

## Data, privacy, and operating limits

- The dashboard is labelled as using **fictionalised assignment data**. It is a demonstration tool, not a live market-data service.
- The priority score is rule-based and should not be interpreted as investment advice.
- Semantic search runs using an in-browser model and a precomputed local index; first-time model download requires an internet connection.
- Ask the Feed sends the user's question and the selected records' canonical fields to Cloudflare Workers AI for generation. Do not use it for confidential research unless the dataset and provider configuration are approved for that use.
- Cloudflare credentials must remain server-side and must be stored in local `.env.local` for development and Vercel environment variables for production.
- Cloudflare Workers AI has usage limits that depend on the account plan. Check the current Cloudflare pricing/usage page before public release; free usage is quota-limited, not unlimited.
- The API route has a best-effort in-memory request limit intended for a small demo. It does not replace authentication, bot protection, or a globally shared rate-limit store.
- Similarity clustering and retrieval are heuristic systems. Always inspect the underlying records and citations when accuracy matters.

---

## Repository structure

```text
src/
  app/
    api/
      ask-feed/
        route.ts          # Server-side retrieval verification and Cloudflare AI call
    page.tsx              # Research UI, filters, search, clustering, Q&A, and reading pane
    layout.tsx            # App layout and font configuration
scripts/
  generate-embeddings.mjs # Precomputes record vectors for semantic retrieval
public/
  data/
    Aequitas_Research_Intelligence_Assignment_Data/
      research_feed.csv
      company_master.csv
      watchlist.csv
      research_embeddings.json
      raw_documents/
```

---

## License and attribution

The project uses the assignment's supplied fictionalised research dataset. Check the terms that accompanied any separately downloaded model or library before redistributing those assets. Add or update a repository `LICENSE` file if you intend to license the application code publicly.
