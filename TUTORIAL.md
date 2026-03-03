# 🚀 Build Your First Trading RAG Mentor: Step-by-Step Tutorial

Welcome! By the end of this tutorial, you will have your very own fully-functional, locally-running AI Trading Mentor based on Retrieval-Augmented Generation (RAG). You'll learn how to take standard text or PDF documents (your trading transcripts, rules, notes), convert them into searchable math vectors, and hook them up to a Large Language Model (LLM) so you can literally chat with your data. This is the foundation of modern enterprise AI.

---

## 🛠️ Prerequisites

Before we start, gather these tools:

- **Python 3.9+** and standard `pip`
- **Git** (to clone the repo)
- **`jq`** installed (`brew install jq` on macOS or `apt install jq` on Linux) for parsing JSON documents
- **Groq API Key**: Get a free, lightning-fast API key at [console.groq.com](https://console.groq.com)

---

## 🤔 Why is RAG Required?

### The Problem With Pure LLMs

Large Language Models (LLMs) like LLaMA or Gemini are brilliant generalists — they have absorbed vast amounts of public text during training. But they have three critical weaknesses that make them unreliable for domain-specific, private, or always-current knowledge:

| Problem                   | What Happens In Practice                                                                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge Cutoff**      | The LLM was trained on data up to a fixed date. It knows nothing about your personal trading transcripts, your specific setups, or recent market events.  |
| **Hallucination**         | When the LLM doesn't know an answer, it confidently _makes one up_. For trading decisions, a hallucinated rule could mean real financial loss.            |
| **No Private Data**       | Your trading journal, video transcripts, and personal notes never go into a public LLM's training. Without RAG, the LLM can never learn _your_ style.     |
| **Context Window Limits** | Feeding thousands of pages of transcripts directly into an LLM prompt is technically impossible — the context window is too small, and it gets expensive. |

### Why RAG Solves This

RAG (Retrieval-Augmented Generation) solves all four problems:

- ✅ **It reads _your_ documents at query time** — not at training time — so it always has access to the latest content you upload.
- ✅ **It grounds the LLM's answer in retrieved excerpts**, making hallucinations near-impossible because the model is told to answer _only_ from what it retrieved.
- ✅ **It handles private data** — your transcripts never need to be part of any model's training.
- ✅ **It's efficient** — instead of feeding all documents to the LLM, only the most relevant 5–10 chunks (paragraphs) are sent.

> **In this project:** The prompt in `src/prompts.py` explicitly instructs the LLM:  
> _"Use ONLY the exact information and wording present in the provided excerpts. NEVER add, infer, generalize, assume... or use any external knowledge."_  
> This is RAG grounding in action.

---

## 💡 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances a Large Language Model by first _retrieving_ the most relevant pieces of information from a knowledge base, and then _augmenting_ the LLM's prompt with that retrieved context before generating a response.

Think of it as giving your LLM an open-book exam, where the "book" is your personal collection of trading documents.

### The Core Idea — Three Phases

```
Your Question
     │
     ▼
┌────────────┐      ┌──────────────────────────────┐
│  RETRIEVE  │─────▶│  Your Knowledge Base          │
│            │◀─────│  (ChromaDB + BM25 Index)      │
└────────────┘      └──────────────────────────────┘
     │
     │  Top-K most relevant chunks (re-ranked)
     ▼
┌────────────┐
│  AUGMENT   │  ← Inject retrieved chunks into the prompt
│            │    (src/prompts.py TRADING_MENTOR_PROMPT)
└────────────┘
     │
     ▼
┌────────────┐      ┌──────────────────────────────┐
│  GENERATE  │─────▶│  Groq LLM (LLaMA-3.3-70b)   │
│            │◀─────│  or Ollama (local)            │
└────────────┘      └──────────────────────────────┘
     │
     ▼
 Final Answer (grounded in your transcripts)
```

### A Concrete Example

> **You ask:** _"What is the volume rule for a VCP pattern?"_  
> **Without RAG:** The LLM guesses from public knowledge — potentially wrong or generic.  
> **With RAG:** The system finds the exact chunk from your `vcp_rules.txt` transcript that talks about volume, injects it into the prompt, and the LLM answers _only from that text_.

---

## 🧩 Main Components of RAG

A production RAG system has **five core components**. Here is how each maps to a specific file or tool in this project:

| #   | Component           | Role                                        | Tool / File in This Project                                           |
| --- | ------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| 1   | **Document Loader** | Read raw files (`.txt`, `.pdf`, `.json`)    | `LangChain` loaders in `src/build_index.py`                           |
| 2   | **Text Splitter**   | Chunk long documents into digestible pieces | `RecursiveCharacterTextSplitter` in `src/build_index.py`              |
| 3   | **Embedding Model** | Convert text chunks into math vectors       | `HuggingFace all-MiniLM-L6-v2` in `src/build_index.py`                |
| 4   | **Vector Store**    | Store and search vectors efficiently        | `ChromaDB` (persisted in `data/chroma_db/`)                           |
| 5   | **Retriever + LLM** | Find relevant chunks and generate an answer | `BM25 + CrossEncoder + Groq LLM` in `src/retrieval.py` & `src/app.py` |

Two bonus components that make this project production-ready:

| #   | Component           | Role                                      | Tool / File                                      |
| --- | ------------------- | ----------------------------------------- | ------------------------------------------------ |
| 6   | **Prompt Template** | Groundings, persona, and answer structure | `src/prompts.py`                                 |
| 7   | **Observability**   | Trace every retrieval and generation call | `LangSmith` (`@traceable` in `src/retrieval.py`) |

---

### 📂 Component 1 — Document Loader

#### What is a Document Loader?

A **Document Loader** is the entry point of any RAG pipeline. Its job is to read raw files from disk (or the web, or a database), parse their content, and convert them into a standard in-memory format that the rest of the pipeline can work with. In LangChain, this standard format is a list of `Document` objects — each containing `page_content` (the raw text) and `metadata` (source path, page number, etc.).

Without a loader, you can't get your data _into_ the pipeline at all.

#### How This Project Uses It

**File:** `src/build_index.py`

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader

if ext == '.txt':
    loader = TextLoader(str(abs_path))
elif ext == '.pdf':
    loader = PyPDFLoader(str(abs_path))
elif ext == '.json':
    loader = JSONLoader(str(abs_path), jq_schema='.', text_content=False)
```

The project auto-detects file extension and picks the right loader. The `jq_schema='.'` on JSONLoader means it flattens the entire JSON tree into text — useful for structured trade logs.

#### Full Landscape of Available Loaders

| Loader                  | Best For                                              | Library               |
| ----------------------- | ----------------------------------------------------- | --------------------- |
| `TextLoader`            | Plain `.txt` files                                    | `langchain_community` |
| `PyPDFLoader`           | Standard PDFs with text layers                        | `langchain_community` |
| `PDFPlumberLoader`      | PDFs with complex tables & images                     | `langchain_community` |
| `UnstructuredPDFLoader` | Scanned PDFs using OCR                                | `langchain_community` |
| `JSONLoader`            | JSON files with `jq` path selectors                   | `langchain_community` |
| `CSVLoader`             | Spreadsheets and tabular data                         | `langchain_community` |
| `NotionDirectoryLoader` | Exported Notion pages                                 | `langchain_community` |
| `WebBaseLoader`         | Scrape live web pages                                 | `langchain_community` |
| `YoutubeLoader`         | YouTube video transcripts via API                     | `langchain_community` |
| `DocugamiLoader`        | Enterprise Word/XML documents                         | `langchain_community` |
| `UnstructuredLoader`    | Any format (docx, pptx, html, email)                  | `unstructured`        |
| `LlamaParseLoader`      | Premium AI-first PDF parser (handles tables, figures) | `llama_parse`         |

#### Why This Choice?

- **`TextLoader`** — Trading transcripts exported from video captions or notes are almost always plain text. It's the simplest, most reliable, and zero-dependency option.
- **`PyPDFLoader`** — PDFs are the most common format for brokerage statements, research reports, and saved web pages. PyPDF is lightweight, widely tested, and handles standard text-layer PDFs perfectly.
- **`JSONLoader`** — Trade journals or alert systems often export structured JSON. Using `jq_schema` gives you fine-grained control over exactly which fields to extract.

#### What You Could Try Instead

| Alternative          | When to Consider                                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `PDFPlumberLoader`   | Your PDFs have multi-column layouts or tables (e.g., earnings reports with financial tables)                                                 |
| `UnstructuredLoader` | You want to drop in Word docs (`.docx`), PowerPoints, HTML exports without format-specific code                                              |
| `LlamaParseLoader`   | You have complex PDFs with charts, scanned images, or mixed layouts — LlamaParse uses AI to extract structured text with far higher accuracy |
| `YoutubeLoader`      | You have YouTube trading video links — this can auto-download transcripts and feed them directly into the pipeline                           |

---

### ✂️ Component 2 — Text Splitter

#### What is a Text Splitter?

A **Text Splitter** breaks long documents into smaller, overlapping pieces called **chunks**. This is necessary because:

1. **LLM context windows** have a maximum size — you can't send an entire 200-page trading book as context.
2. **Embedding quality degrades** on long texts — short, focused chunks produce more precise vector representations.
3. **Retrieval precision improves** — retrieving a 200-word chunk about "VCP entry criteria" is far more useful than retrieving a 5,000-word chapter.

The **overlap** ensures that a sentence split across chunk boundaries is still findable.

#### How This Project Uses It

**File:** `src/build_index.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
```

- **`chunk_size=800`** — Each chunk is up to 800 characters (~120–150 words). This fits comfortably within the embedding model's 256-token limit and gives the LLM a tight, focused excerpt.
- **`chunk_overlap=100`** — A 100-character rolling overlap prevents context loss at split boundaries. If a VCP entry rule starts at character 790 of a chunk, it won't be missed.
- **`RecursiveCharacterTextSplitter`** — Tries to split at paragraph breaks (`\n\n`) first, then sentence breaks (`\n`), then word boundaries, and only as a last resort mid-word. This preserves semantic coherence as much as possible.

#### Full Landscape of Available Splitters

| Splitter                                | Strategy                                     | Best For                                         |
| --------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| `CharacterTextSplitter`                 | Splits at a single separator character       | Simple, predictable splits                       |
| `RecursiveCharacterTextSplitter`        | Tries multiple separators in order           | Most general-purpose text ✅                     |
| `TokenTextSplitter`                     | Splits by token count (not characters)       | When you need exact token control for LLM limits |
| `MarkdownHeaderTextSplitter`            | Splits at markdown `#` headings              | Structured markdown documents / wikis            |
| `HTMLHeaderTextSplitter`                | Splits at HTML `<h1>`, `<h2>` tags           | Web scraped content                              |
| `SentenceTransformersTokenTextSplitter` | Token-aware + embedding-model-aware          | When embedding model is the bottleneck           |
| `SemanticChunker`                       | Uses embeddings to find natural topic breaks | Preserving topic coherence over character count  |
| `SpacyTextSplitter`                     | Splits at sentence boundaries via NLP        | When grammatical sentence integrity matters      |

#### Why This Choice?

`RecursiveCharacterTextSplitter` is the **de-facto standard** for general-purpose text for good reason:

- Trading transcripts are unstructured, conversational text — they don't always have clean headings or paragraphs.
- The recursive strategy respects natural sentence and paragraph flow wherever possible.
- 800 characters is a well-tested sweet spot — large enough to contain a complete trading rule with context, small enough to remain precise when retrieved.

#### What You Could Try Instead

| Alternative                        | When to Consider                                                                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TokenTextSplitter`                | If you switch your embedding model to one with a strict token limit (e.g., OpenAI `text-embedding-3` with 8191 tokens max)                                |
| `MarkdownHeaderTextSplitter`       | If you store your trading notes as structured markdown files with section headers — retrieval becomes topic-aware                                         |
| `SemanticChunker`                  | For highest quality — it uses embeddings to detect natural topic transitions instead of cutting blindly at character 800. Trade-off: much slower indexing |
| Increase `chunk_size` to 1200–2000 | If you find answers are getting cut off mid-explanation. Monitor retrieval quality with `scripts/evaluate_rag.py`                                         |

---

### 🔢 Component 3 — Embedding Model

#### What is an Embedding Model?

An **Embedding Model** converts text into a list of numbers called a **vector** (or embedding). The key property is that semantically similar texts produce vectors that are mathematically close to each other in vector space.

For example:

- `"VCP requires volume to dry up"` → `[0.23, -0.11, 0.87, ...]` (384 numbers)
- `"Volatility Contraction needs declining volume"` → `[0.25, -0.09, 0.84, ...]` (very close!)
- `"What is the RSI formula?"` → `[0.91, 0.44, -0.23, ...]` (far away)

This is what makes **semantic search** possible — it finds related content even when the exact words don't match.

#### How This Project Uses It

**File:** `src/build_index.py`

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

- **`all-MiniLM-L6-v2`** is a sentence-transformer model from HuggingFace.
- It produces **384-dimensional vectors**.
- It runs **100% locally on CPU** — no API calls, no cost, no data sent anywhere.
- The model is downloaded once (~90 MB) and cached by HuggingFace locally.

#### Full Landscape of Embedding Models

| Model                             | Dimensions | Cost             | Privacy  | Speed  | Quality             |
| --------------------------------- | ---------- | ---------------- | -------- | ------ | ------------------- |
| `all-MiniLM-L6-v2` (HuggingFace)  | 384        | Free             | 🟢 Local | Fast   | Good ✅             |
| `all-mpnet-base-v2` (HuggingFace) | 768        | Free             | 🟢 Local | Medium | Better              |
| `bge-large-en-v1.5` (HuggingFace) | 1024       | Free             | 🟢 Local | Slow   | Best local          |
| `nomic-embed-text` (Ollama)       | 768        | Free             | 🟢 Local | Fast   | Good                |
| `text-embedding-3-small` (OpenAI) | 1536       | ~$0.02/1M tokens | 🔴 Cloud | Fast   | Excellent           |
| `text-embedding-3-large` (OpenAI) | 3072       | ~$0.13/1M tokens | 🔴 Cloud | Fast   | State-of-Art        |
| `embed-english-v3.0` (Cohere)     | 1024       | Paid             | 🔴 Cloud | Fast   | Excellent           |
| `voyage-finance-2` (VoyageAI)     | 1024       | Paid             | 🔴 Cloud | Fast   | Best for finance 🏆 |

#### Why This Choice?

- **Free and local**: The project is designed for individual traders and learners with zero cloud costs. `all-MiniLM-L6-v2` is entirely on-CPU with no API key required.
- **Battle-tested**: It's the most widely benchmarked sentence-transformer model in the world, with strong MTEB scores for semantic similarity tasks.
- **Fast enough**: For a corpus of hundreds of trading transcripts, indexing completes in seconds on CPU.
- **Good enough quality**: For domain-specific trading text where the vocabulary is relatively consistent (VCP, breakout, momentum, etc.), 384-dimensional semantic similarity performs extremely well.

#### What You Could Try Instead

| Alternative                       | Expected Improvement                                                       | Trade-off                                        |
| --------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| `all-mpnet-base-v2`               | +5–10% semantic accuracy                                                   | 2× slower, uses 768 dims (more ChromaDB storage) |
| `bge-large-en-v1.5`               | Best local quality (SOTA on MTEB)                                          | ~3× slower, 1024 dims                            |
| `voyage-finance-2` (VoyageAI)     | Specifically trained on financial text — likely the best for this use case | Paid API, data leaves your machine               |
| `nomic-embed-text` via Ollama     | Good quality, still fully local                                            | Must have Ollama installed and running           |
| `text-embedding-3-small` (OpenAI) | Excellent general quality                                                  | Costs money, sends data to OpenAI                |

> **Pro tip:** If you change the embedding model, you **must re-index everything from scratch** — the new model's vector space is incompatible with the old one. Delete `data/chroma_db/` and run `python src/build_index.py`.

---

### 🗄️ Component 4 — Vector Store

#### What is a Vector Store?

A **Vector Store** (also called a vector database) is a specialized database designed to store high-dimensional vectors and answer one key query: _"Given this query vector, find me the K most similar vectors in the database."_ It does this via Approximate Nearest Neighbor (ANN) search algorithms, which are vastly faster than comparing against every stored vector one by one.

In a RAG system, the vector store is the heart of semantic retrieval. It stores your embedded document chunks at index time and retrieves the closest ones at query time.

#### How This Project Uses It

**File:** `src/build_index.py`, `src/app.py`

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
vectorstore.add_documents(chunks)

# At query time in retrieval.py:
results = vectorstore.similarity_search_with_relevance_scores(query, k=12)
```

- **ChromaDB** is stored locally in `data/chroma_db/` as a set of SQLite and Parquet files.
- It persists across restarts — no need to re-embed every time the app starts.
- Incremental updates: the project uses `source` metadata to delete stale chunks before re-adding, so only changed files trigger re-embedding.
- Returns relevance scores alongside documents for use in the UI's badge display.

#### Full Landscape of Vector Stores

| Vector Store         | Deployment            | Best For                                    | Scale             |
| -------------------- | --------------------- | ------------------------------------------- | ----------------- |
| **ChromaDB**         | Local (embedded)      | Local dev, small–medium corpora             | Up to ~1M docs ✅ |
| **FAISS** (Facebook) | Local (in-memory)     | Maximum speed, research                     | Up to ~10M docs   |
| **LanceDB**          | Local (embedded)      | Disk-efficient, columnar storage            | Medium–large      |
| **Weaviate**         | Self-hosted / Cloud   | Multi-modal search, rich metadata filtering | Large scale       |
| **Pinecone**         | Cloud (fully managed) | Production SaaS, zero ops                   | Unlimited         |
| **Qdrant**           | Self-hosted / Cloud   | Payload filtering, fast ANN                 | Large scale       |
| **Milvus**           | Self-hosted           | Enterprise scale, GPU acceleration          | Very large        |
| **pgvector**         | PostgreSQL extension  | Already using PostgreSQL                    | Small–medium      |
| **OpenSearch**       | Self-hosted / AWS     | Hybrid search in existing ES clusters       | Large             |

#### Why This Choice?

- **Zero infrastructure**: ChromaDB runs entirely in-process with no server, no Docker, no cloud account. Perfect for a local trading mentor app.
- **Persistence**: Data survives restarts without re-embedding — critical because embedding ~90MB worth of transcripts takes time.
- **LangChain native**: First-class integration means one-line setup.
- **Incremental updates**: The project's file-hash system + ChromaDB's `delete(where={"source": ...})` allows surgical updates without full rebuilds.
- **Sufficient scale**: A personal trading knowledge base will have tens of thousands of chunks at most — well within ChromaDB's sweet spot.

#### What You Could Try Instead

| Alternative  | When to Consider                                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FAISS**    | If you need maximum query speed (microsecond latency) for a large static corpus. FAISS is in-memory only — you'd need to serialize/deserialize it.                      |
| **LanceDB**  | More disk-efficient than ChromaDB for large corpora; also supports multi-modal (images + text). Good drop-in replacement.                                               |
| **Qdrant**   | If you want to filter by metadata at query time (e.g., "only search transcripts from 2024") — Qdrant's payload filters are significantly more powerful than ChromaDB's. |
| **Pinecone** | If you deploy this as a multi-user SaaS product and need managed cloud storage with zero DevOps.                                                                        |
| **pgvector** | If you're already using PostgreSQL for other app data — consolidates your storage to one database.                                                                      |

---

### 🔍 Component 5 — Retriever + LLM

#### What is a Retriever?

A **Retriever** is the component that, given a user query, fetches the most relevant document chunks from the knowledge base. It bridges the query and the vector store. While "simple" vector similarity retrieval asks "which chunks are semantically closest?", advanced retrieval strategies layer multiple techniques to maximize both _recall_ (finding all relevant chunks) and _precision_ (discarding irrelevant ones).

#### What is an LLM (in a RAG context)?

The **LLM (Large Language Model)** is the final generation step. After retrieval, the LLM receives the user's question _plus_ the retrieved chunks as context, and produces a fluent, synthesized answer. In RAG it does **not** need to memorize facts — it only needs to read, summarize, and reason over the provided excerpts.

#### How This Project Uses It — A 3-Stage Retrieval Pipeline

This project implements one of the most sophisticated retrieval pipelines available. **File:** `src/retrieval.py`

**Stage 0: Query Reformulation** (`src/app.py` + `REFORMULATION_PROMPT` in `src/prompts.py`)

```
"How does volume play into that?" → "How does volume play into the VCP pattern?"
```

Ambiguous follow-up questions are rewritten into standalone queries using the LLM itself, ensuring history context doesn't break retrieval.

**Stage 1: Hybrid Retrieval** (`hybrid_retrieve` function)

```python
# BM25 (keyword) + Vector (semantic) → interleaved merge at fetch_k = k × 2
bm25_docs = bm25_retriever.invoke(query)          # keyword match
vector_results = vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)  # semantic match
# → interleaved merge → 12 candidate chunks
```

**Stage 2: Cross-Encoder Re-ranking** (`rerank_documents` function)

```python
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
ce_scores = cross_encoder.predict([(query, chunk) for chunk in candidates])
# → top 6 highest-scoring chunks sent to LLM
```

**LLM Generation** (`src/app.py`)

```python
# Groq API with llama-3.3-70b-versatile (default)
# or Ollama with gemma3:1b (local private)
```

#### Full Landscape of Retrieval Strategies

| Strategy                                   | How It Works                                    | Strength                     |
| ------------------------------------------ | ----------------------------------------------- | ---------------------------- |
| **Pure Vector Search**                     | Cosine similarity on embeddings                 | Semantic understanding       |
| **BM25 / Keyword**                         | TF-IDF statistical relevance                    | Exact term matching          |
| **Hybrid (BM25 + Vector)**                 | Merge both result sets                          | Best of both worlds ✅       |
| **Cross-Encoder Re-ranking**               | Joint query+doc scoring                         | Highest precision ✅         |
| **MMR (Max Marginal Relevance)**           | Penalizes redundant chunks                      | Diversity in results         |
| **Multi-Query Retrieval**                  | LLM generates N rephrasings of query            | Higher recall                |
| **HyDE (Hypothetical Document Embedding)** | LLM writes a hypothetical answer, then searches | Bridges query-document gap   |
| **Self-RAG**                               | LLM decides whether to retrieve at all          | Avoids unnecessary retrieval |
| **Contextual Compression**                 | LLM trims chunks to only relevant sentences     | Cleaner context window       |
| **Ensemble Retriever**                     | Weighted merge of N arbitrary retrievers        | Maximum flexibility          |

#### Full Landscape of LLMs Available

| LLM                       | Provider          | Privacy    | Speed       | Quality      | Cost             |
| ------------------------- | ----------------- | ---------- | ----------- | ------------ | ---------------- |
| `llama-3.3-70b-versatile` | Groq (cloud)      | 🔴 Cloud   | ⚡⚡⚡      | Excellent    | Free tier ✅     |
| `llama-3.1-8b-instant`    | Groq (cloud)      | 🔴 Cloud   | ⚡⚡⚡⚡    | Good         | Free tier        |
| `mixtral-8x7b-32768`      | Groq (cloud)      | 🔴 Cloud   | ⚡⚡⚡      | Very Good    | Free tier        |
| `gemma3:1b`               | Ollama (local)    | 🟢 Private | Fast on CPU | Decent       | Free             |
| `llama3.2:3b`             | Ollama (local)    | 🟢 Private | Fast on CPU | Good         | Free             |
| `mistral:7b`              | Ollama (local)    | 🟢 Private | Medium      | Good         | Free             |
| `gpt-4o`                  | OpenAI (cloud)    | 🔴 Cloud   | Fast        | State-of-Art | ~$5–15/1M tokens |
| `claude-3-5-sonnet`       | Anthropic (cloud) | 🔴 Cloud   | Fast        | State-of-Art | ~$3/1M tokens    |
| `gemini-1.5-pro`          | Google (cloud)    | 🔴 Cloud   | Fast        | Excellent    | ~$3.5/1M tokens  |

#### Why This Choice?

**Retrieval (BM25 + Vector + CrossEncoder):**

- Trading has many **exact acronyms and tickers** (VCP, MACD, EMA, SPY) — pure vector search misses exact-match needs; BM25 fills this gap perfectly.
- Cross-encoder re-ranking dramatically reduces false positives where a chunk scores high on vector similarity but isn't actually relevant to the specific question asked.
- Together, this 3-stage pipeline approaches the quality of paid retrieval services.

**LLM (Groq llama-3.3-70b):**

- Groq provides **inference at hundreds of tokens/second** — faster than any other cloud provider — making the app feel instant despite using a 70B parameter model.
- The free tier is generous enough for hundreds of queries per day, making it ideal for personal use.
- 70B parameters gives near-GPT-4-class reasoning quality for synthesizing trading rules from context.

#### What You Could Try Instead

| Alternative                             | Expected Improvement                                                                       | Trade-off                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| **Multi-Query Retrieval**               | Higher recall (generates 3–5 question rephrasings)                                         | 3–5× more LLM calls during retrieval     |
| **HyDE Retrieval**                      | Better for abstract questions — LLM first imagines a concrete answer, then searches for it | Extra LLM call per query                 |
| **MMR Retrieval**                       | More diverse results — prevents retrieving the same paragraph 3 times                      | May reduce relevance in favor of variety |
| **Contextual Compression**              | Shrinks retrieved chunks to only relevant sentences, saving LLM context window             | Extra LLM call per chunk                 |
| **GPT-4o or Claude-3.5 as LLM**         | Best-in-class reasoning and instruction following                                          | Costs money per query                    |
| **Smaller Groq model (`llama-3.1-8b`)** | 3× faster responses                                                                        | Notably weaker at multi-step reasoning   |

---

### 📝 Component 6 — Prompt Template

#### What is a Prompt Template?

A **Prompt Template** is the structured instruction you give the LLM _every single time_ it's called. It defines the AI's persona, its constraints, exactly where to inject retrieved context and chat history, and the expected format of its answer. The quality of your prompt is often the **biggest lever** for improving RAG answer quality — even more than the retrieval step itself.

A good prompt template functions like an employee onboarding manual: it tells the model who it is, what it can and cannot do, and how to format its output.

#### How This Project Uses It

**File:** `src/prompts.py` — two templates are defined:

**1. `TRADING_MENTOR_PROMPT`** — The main generation prompt. It has four critical sections:

```
├── Persona definition:    "You are my strict, no-nonsense Personal Trading Mentor..."
├── Ultra-strict rules:    "Use ONLY exact information from excerpts. NEVER infer..."
├── RAG fallback rule:     "If unanswerable, say 'I am a RAG assistant...'"
├── Answer structure:      1. Direct Answer → 2. Reasoning → 3. Risk → 4. Notes → 5. Action
└── Variables:
    ├── {context}          ← the retrieved document chunks (injected by retrieval step)
    ├── {chat_history}     ← last N turns from SQLite
    └── {question}         ← the user's reformulated query
```

**2. `REFORMULATION_PROMPT`** — A meta-prompt used to rewrite ambiguous follow-up questions into standalone ones before retrieval fires. This ensures retrieval quality is never degraded by pronouns like "that" or "it".

#### Types of Prompt Engineering Strategies

| Strategy                      | What It Does                                                        |
| ----------------------------- | ------------------------------------------------------------------- |
| **Zero-shot**                 | Just ask the question — no examples                                 |
| **Few-shot**                  | Provide 2–3 examples of good Q&A pairs before the question          |
| **Chain-of-Thought (CoT)**    | Instruct the model to "think step by step" before answering         |
| **RAG Grounding**             | Inject retrieved context + strict "only use this" instructions ✅   |
| **ReAct (Reason + Act)**      | LLM reasons, then calls tools, then reasons again in a loop         |
| **System / User / Assistant** | Standard multi-turn chat message format for chat models             |
| **Constitutional AI**         | Self-critiquing prompts that ask the model to revise its own answer |

#### Why This Choice?

- **Strict grounding rules** eliminate hallucination — the single biggest trust problem with LLMs in trading (where wrong rules cost real money).
- **Persona + structure** make answers consistently useful — new traders get step-by-step formatted answers; experts get direct answers without fluff.
- **`REFORMULATION_PROMPT`** is a hallmark of production RAG — without it, multi-turn conversations progressively degrade retrieval quality.
- The fallback message explicitly tells the user _why_ the system can't answer (no real-time data, answers only from transcripts), which builds trust instead of confusing the user.

#### What You Could Try Instead

| Alternative                                          | Expected Improvement                                                                                                |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Add few-shot examples** to `TRADING_MENTOR_PROMPT` | More consistent answer formatting, better instruction following                                                     |
| **Chain-of-Thought instructions**                    | Better multi-step reasoning (e.g., "First identify the setup, then identify the entry rule...")                     |
| **Separate system prompt** from user prompt          | Cleaner prompt structure for providers that support system/user roles (e.g., OpenAI API)                            |
| **Language-specific prompts**                        | Add instructions for specific trading strategies: "When answering about momentum, always mention risk/reward ratio" |

---

### 🔭 Component 7 — Observability (LangSmith)

#### What is Observability in RAG Systems?

**Observability** means having full visibility into what your RAG pipeline does at each step during a real user query. Without it, when the system gives a wrong answer, you can only guess whether the problem was:

- 🤔 Wrong chunks retrieved?
- 🤔 Chunk retrieved but ranked poorly by the cross-encoder?
- 🤔 Good chunks retrieved but the prompt failed to use them?
- 🤔 LLM hallucinated despite good context?

Observability makes this **pinpoint-diagnosable** instead of a debugging nightmare.

#### How This Project Uses It

**File:** `src/retrieval.py` — every retrieval function is decorated with `@traceable` from LangSmith:

```python
from langsmith import traceable

@traceable(run_type="retriever", name="build_bm25_retriever")
def build_bm25_retriever(...): ...

@traceable(run_type="retriever", name="hybrid_retrieve_stage_1")
def hybrid_retrieve(...): ...

@traceable(run_type="tool", name="cross_encoder_rerank")
def rerank_documents(...): ...

@traceable(run_type="retriever", name="hybrid_retrieve_and_rerank")
def hybrid_retrieve_and_rerank(...): ...
```

This automatically sends a full execution trace to the **LangSmith dashboard** at [smith.langchain.com](https://smith.langchain.com), capturing:

- Input query and reformulated query
- Which documents were retrieved at Stage 1 (BM25 + Vector) and their initial scores
- Which documents survived re-ranking and their final cross-encoder scores
- The assembled prompt sent to the LLM
- The LLM response and latency
- Token usage

Configuration in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=trading-rag-mentor
```

#### Full Landscape of RAG Observability Tools

| Tool                       | Focus                                                    | Deployment                |
| -------------------------- | -------------------------------------------------------- | ------------------------- |
| **LangSmith**              | LangChain-native tracing, evaluation, dataset management | Cloud (LangChain Inc.) ✅ |
| **LangFuse**               | Open-source tracing + scoring + LLM analytics            | Self-hosted or Cloud      |
| **Arize Phoenix**          | ML observability, embedding drift detection              | Local or Cloud            |
| **Weights & Biases (W&B)** | Experiment tracking, prompt versioning                   | Cloud                     |
| **PromptLayer**            | Prompt version control + usage analytics                 | Cloud                     |
| **Helicone**               | LLM request logging, cost tracking                       | Cloud proxy               |
| **OpenTelemetry**          | Standard distributed tracing (DIY)                       | Self-hosted               |
| **Custom logging**         | `logging` module + JSON logs                             | Fully local               |

#### Why This Choice?

- **LangChain-native**: Since this project uses LangChain for everything, adding `@traceable` is literally a one-line decorator — zero plumbing required.
- **End-to-end visibility**: LangSmith traces the entire chain from user question → retrieval → prompt → LLM → answer, not just isolated steps.
- **Dataset builder**: Failed queries can be saved directly as evaluation datasets in LangSmith, creating a feedback loop for improving the system.
- **Free tier**: Sufficient for personal use with thousands of traces per month.

#### What You Could Try Instead

| Alternative          | When to Consider                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **LangFuse**         | If you want **100% self-hosted** observability with no third-party cloud — LangFuse runs on your own server via Docker    |
| **Arize Phoenix**    | If you want to analyze **embedding drift** — detect when your knowledge base chunks are becoming poorly matched over time |
| **Custom `logging`** | Simplest option: add structured JSON logging to every function. No external service, but no dashboard either              |
| **Weights & Biases** | If you run systematic prompt experiments and want to track which prompt version gave the best evaluation scores           |

---

## 🏗️ RAG Architecture — Deep Dive

This project uses a **two-phase, five-layer architecture**. Here is a detailed breakdown of every layer and the tool powering it.

### Phase 1: Indexing Pipeline (Offline — runs once or on Re-index)

This phase converts your raw documents into a searchable vector database. It is triggered by running `python src/build_index.py` or clicking **🔄 Re-index** in the sidebar.

```
data/transcripts/         ← Your source of truth
  ├── vcp_rules.txt
  ├── momentum_notes.pdf
  └── trade_journal.json
        │
        ▼
┌─────────────────────────────────────────────────┐
│  LAYER 1 — Document Loaders (src/build_index.py) │
│                                                   │
│  .txt  → TextLoader (LangChain)                   │
│  .pdf  → PyPDFLoader (LangChain)                  │
│  .json → JSONLoader with jq_schema (LangChain)    │
│                                                   │
│  MD5 hashing tracks changed files                 │
│  → Only new/modified files are re-processed       │
└────────────────────┬────────────────────────────┘
                     │ LangChain Document objects
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2 — Text Splitter (src/build_index.py)    │
│                                                   │
│  RecursiveCharacterTextSplitter                   │
│    chunk_size   = 800 characters                  │
│    chunk_overlap = 100 characters                 │
│                                                   │
│  Splits long documents into smaller chunks        │
│  so the LLM receives focused, relevant excerpts   │
└────────────────────┬────────────────────────────┘
                     │ List of text chunks (Documents)
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 3 — Embedding Model (src/build_index.py)  │
│                                                   │
│  HuggingFaceEmbeddings("all-MiniLM-L6-v2")       │
│  • Runs 100% locally on your CPU (free, private)  │
│  • Converts each chunk → 384-dimension vector     │
│  • Captures semantic meaning, not just keywords   │
│                                                   │
│  "VCP setup" and "Volatility Contraction"         │
│   both map to similar vector coordinates          │
└────────────────────┬────────────────────────────┘
                     │ 384-dim float vectors + metadata
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 4 — Vector Store (ChromaDB)               │
│                                                   │
│  Persisted to: data/chroma_db/                    │
│  • Stores vectors + original text + metadata      │
│  • Supports cosine similarity search              │
│  • Incremental: deletes stale chunks by           │
│    `source` metadata before re-adding             │
└─────────────────────────────────────────────────┘
```

---

### Phase 2: Query Pipeline (Online — runs on every user question)

This phase happens in real time when you ask a question in the chat UI or API. It is orchestrated from `src/app.py` and `src/retrieval.py`.

```
User Question: "What is the volume rule for a VCP pattern?"
        │
        ▼
┌─────────────────────────────────────────────────┐
│  LAYER 0 — Query Reformulation (src/app.py)      │
│                                                   │
│  REFORMULATION_PROMPT (src/prompts.py)            │
│  • Rewrites ambiguous follow-up questions into    │
│    standalone questions (e.g., "How does volume   │
│    play into that?" → "How does volume play into  │
│    the VCP pattern?")                             │
│  • Ensures conversation context is preserved      │
└────────────────────┬────────────────────────────┘
                     │ Standalone query string
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 1 — Stage 1: Hybrid Retrieval             │
│            (src/retrieval.py → hybrid_retrieve)   │
│                                                   │
│  ┌─────────────────┐   ┌─────────────────────┐   │
│  │  Vector Search  │   │   BM25 Search        │   │
│  │  (ChromaDB)     │   │   (BM25Retriever)    │   │
│  │                 │   │                     │   │
│  │  Semantic match │   │  Keyword/exact match │   │
│  │  "VCP setup"    │   │  Finds "VCP" exactly │   │
│  │  ≈ "Volatility  │   │  even without meaning│   │
│  │  Contraction"   │   │  similarity          │   │
│  └────────┬────────┘   └──────────┬──────────┘   │
│           └────────────┬──────────┘              │
│                        │  Interleaved merge       │
│                        │  (fetch_k = k × 2 = 12) │
└────────────────────────┼────────────────────────┘
                         │ 12 candidate chunks
                         ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2 — Stage 2: Cross-Encoder Re-ranking     │
│            (src/retrieval.py → rerank_documents)  │
│                                                   │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2     │
│  • Takes each (query, chunk) pair                 │
│  • Scores them jointly for relevance              │
│  • Eliminates false positives from Stage 1        │
│  • Returns top-k=6 highest-scoring chunks         │
│                                                   │
│  UI: 🟢 Green badge  = score > 0.5 (high)         │
│       🟡 Yellow badge = 0.2–0.5 (medium)          │
│       🔴 Red badge    = < 0.2 (low)               │
└────────────────────┬────────────────────────────┘
                     │ Top 6 re-ranked chunks + scores
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 3 — Prompt Augmentation (src/prompts.py)  │
│                                                   │
│  TRADING_MENTOR_PROMPT template fills in:         │
│    {context}      ← the 6 retrieved excerpts      │
│    {chat_history} ← last N conversation turns     │
│    {question}     ← the user's (rewritten) query  │
│                                                   │
│  The prompt strictly instructs the LLM:           │
│    "Use ONLY the exact information in excerpts.   │
│     NEVER infer or use external knowledge."       │
└────────────────────┬────────────────────────────┘
                     │ Fully assembled prompt string
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 4 — LLM Generation (src/app.py)           │
│                                                   │
│  Option A: Groq API (default, fast cloud)         │
│    Model: llama-3.3-70b-versatile                 │
│    Config via: GROQ_API_KEY in .env               │
│                                                   │
│  Option B: Ollama (local & private)               │
│    Model: gemma3:1b (or any pulled model)         │
│    No data leaves your machine                    │
│                                                   │
│  Parameters (configurable in UI sidebar):         │
│    temperature  → controls creativity / accuracy  │
│    k (top docs) → how many chunks to retrieve     │
└────────────────────┬────────────────────────────┘
                     │ Generated response text
                     ▼
┌─────────────────────────────────────────────────┐
│  LAYER 5 — UI + Persistence (src/app.py)         │
│                                                   │
│  Streamlit renders the streamed response          │
│  SQLite (chat_history.db) saves the session       │
│  Source citations shown in "📄 Sources" expander  │
└─────────────────────────────────────────────────┘
```

---

### Observability Layer — LangSmith (Cross-Cutting)

Every function in `src/retrieval.py` is decorated with `@traceable` from LangSmith. This captures a full trace of every query — including which documents were retrieved, what scores they received, and how long each step took — in the LangSmith dashboard.

```python
# From src/retrieval.py
@traceable(run_type="retriever", name="hybrid_retrieve_and_rerank")
def hybrid_retrieve_and_rerank(...):
    ...
```

This is essential for debugging retrieval failures and optimizing chunk size, `k`, and reranking thresholds without guesswork.

---

### Full Component Summary Table

| Layer         | Component        | Library / Tool                     | File                  |
| ------------- | ---------------- | ---------------------------------- | --------------------- |
| Ingestion     | Document Loaders | `LangChain Community`              | `src/build_index.py`  |
| Ingestion     | Text Splitting   | `RecursiveCharacterTextSplitter`   | `src/build_index.py`  |
| Ingestion     | Change Detection | `hashlib` (MD5)                    | `src/build_index.py`  |
| Indexing      | Embedding Model  | `HuggingFace all-MiniLM-L6-v2`     | `src/build_index.py`  |
| Indexing      | Vector Store     | `ChromaDB`                         | `src/build_index.py`  |
| Retrieval     | Keyword Search   | `BM25Retriever` (LangChain)        | `src/retrieval.py`    |
| Retrieval     | Semantic Search  | `ChromaDB similarity_search`       | `src/retrieval.py`    |
| Retrieval     | Re-ranking       | `CrossEncoder ms-marco-MiniLM`     | `src/retrieval.py`    |
| Generation    | Prompt Template  | `PromptTemplate` (LangChain)       | `src/prompts.py`      |
| Generation    | Cloud LLM        | `Groq llama-3.3-70b-versatile`     | `src/app.py`          |
| Generation    | Local LLM        | `Ollama gemma3:1b`                 | `src/app.py`          |
| Memory        | Chat History     | `SQLite` via `src/chat_history.py` | `src/chat_history.py` |
| UI            | Frontend         | `Streamlit`                        | `src/app.py`          |
| Observability | Tracing          | `LangSmith @traceable`             | `src/retrieval.py`    |

---

## 🗺️ Step-by-Step Guide

### Step 1: Clone the Repo & Setup Your Environment

**Goal:** Pull down the code and install the necessary Python libraries.  
**Why it matters:** AI applications rely on specific libraries (like LangChain, ChromaDB, and Streamlit) to handle everything from user interfaces to splitting text into chunks.

**Commands:**

```bash
git clone https://github.com/sudhakarbadugu/trading-rag-mentor.git
cd trading-rag-mentor
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**What you should see / verify:**
A terminal prompt that begins with `(venv)` and a list of installed packages concluding without errors.

**Common problems & fixes:**

- _Problem:_ `python3: command not found`
  _Fix:_ Use `python -m venv venv` instead, depending on how your OS aliases Python.
- _Problem:_ Installation fails on a specific package (like ChromaDB).
  _Fix:_ Ensure you have the build tools installed for your OS (e.g., `xcode-select --install` on Mac, or C++ build tools on Windows).

**Next-checkpoint:** Let's authenticate our LLM.

---

### Step 2: Configure Environment Variables

**Goal:** Securely pass your Groq API key into the RAG pipeline.  
**Why it matters:** Hardcoding passwords/API keys into your code is dangerous. Using a `.env` file ensures your keys stay safe on your local machine and never accidentally get uploaded to GitHub.

**Commands:**

```bash
touch .env
```

Open the `.env` file in your favorite text editor (or use `nano .env`) and add:

```env
GROQ_API_KEY=your_actual_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
LLM_PROVIDER=Groq (Fast Cloud)
```

**What you should see / verify:**
You now have a hidden `.env` file in the root directory containing your API key. Ensure there are no spaces around the `=` sign.

**Common problems & fixes:**

- _Problem:_ You accidentally commit `.env` to Git.
  _Fix:_ This repo's `.gitignore` already protects you, but if you do it elsewhere, immediately revoke/delete the key in the Groq console.

**Next-checkpoint:** Passing data into our system.

---

### Step 3: Document Ingestion (Adding Data)

**Goal:** Provide the raw text that the AI will use to answer your questions.  
**Why it matters:** The RAG system knows _nothing_ beyond what you give it. Your documents are the ground truth.

**Commands:**

```bash
mkdir -p data/transcripts
echo "VCP (Volatility Contraction Pattern) requires volume drying up along with price tightening from left to right." > data/transcripts/vcp_rules.txt
```

**What you should see / verify:**
A new file called `vcp_rules.txt` exists inside the `data/transcripts/` directory.

**Common problems & fixes:**

- _Problem:_ Passing unsupported formats (like raw Word docs).
  _Fix:_ The repo specifically supports `.txt`, `.pdf`, and `.json`. Export other documents to formatting-free `.txt` or `.pdf` first.

**Next-checkpoint:** Converting text to vectors.

---

### Step 4: Indexing (BM25 + Vector)

**Goal:** Chop your documents into chunks, calculate their mathematical meaning (vectors), and store them in a database.  
**Why it matters:** LLMs can't read entire books in seconds. Instead, we chunk the text, find the specific chunks mathematically relevant to the user's question, and hand _only_ those to the LLM.

**Commands:**

```bash
python src/build_index.py
```

**What you should see / verify:**
Terminal output saying: `Detected changes: 1 new... Splitting text into chunks... ✅ Index update complete.` A new hidden folder `data/chroma_db/` will be created. Note: The first time you run this, it will download the HuggingFace `all-MiniLM-L6-v2` embedding model (~90MB).

**Common problems & fixes:**

- _Problem:_ `No documents found to index`
  _Fix:_ Double-check that your files are literally inside `data/transcripts/`.
- _Problem:_ Taking a long time on the first run.
  _Fix:_ This is normal as it downloads the local embedding model. Subsequent runs are near-instantaneous.

**Next-checkpoint:** Interacting with your new data.

---

### Step 5: Start the Web App (Minimal UI Test)

**Goal:** Launch the Streamlit front-end to chat with your data.  
**Why it matters:** A chat UI makes it incredibly easy to test retrieval, view chat history, and adjust settings visually rather than via code.

**Commands:**

```bash
streamlit run src/app.py
```

**What you should see / verify:**
Your browser should automatically open to `http://localhost:8501` displaying the "📈 My Personal Trading RAG Coach" interface.
Test it by typing: `"What is a VCP pattern?"`

**Common problems & fixes:**

- _Problem:_ Port 8501 is already in use.
  _Fix:_ Streamlit will usually pick 8502 automatically, but you can force it using `streamlit run src/app.py --server.port 8502`.
- _Problem:_ LLM connection error.
  _Fix:_ Check that your `GROQ_API_KEY` in `.env` is accurate and that your internet is connected.

**Next-checkpoint:** Under the hood of advanced retrieval.

---

### Step 6: Hybrid Querying & Reranker Config

**Goal:** Understand how the UI toggles combine keyword matching (BM25), Semantic Search (Vector), and Cross-encoder reranking.  
**Why it matters:** Vector search (meaning) and keyword search (exact words like "VCP") both have blind spots. Combining them (Hybrid) and then double-checking the results (Reranking) dramatically reduces hallucination and improves accuracy.

**Commands / action:**
In the running UI sidebar, under the `⚙️ RAG Configuration` section:

1. Ensure **🔀 Hybrid Search (BM25 + Vector)** is ON.
2. Ensure **📊 Cross-Encoder Re-ranking** is ON.
3. Open `src/retrieval.py` in your code editor to see how they are configured behind the scenes (look for the `hybrid_retrieve_and_rerank` function).

**What you should see / verify:**
When you ask a question and click the `📄 Sources` expander under the answer, you'll see green/yellow/red badges indicating relevance scores passed back from the Cross-Encoder.

**Common problems & fixes:**

- _Problem:_ First query is slow.
  _Fix:_ The Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) is downloading locally. This only happens once.

**Next-checkpoint:** Changing the AI's personality.

---

### Step 7: Customizing the Prompt Template

**Goal:** Tell the AI _how_ you want it to behave and respond.  
**Why it matters:** The Prompt Template is the instruction manual you give to the LLM right before it answers. By tweaking it, you can make the bot sound academic, funny, or restrict it strictly to bullet points.

**Commands / code:**
Open `src/prompts.py` and locate `TRADING_MENTOR_PROMPT`. Change the first line to:

```python
TRADING_MENTOR_PROMPT = """
You are a highly skeptical, risk-averse trading mentor.
Always warn the user about losing money.
...
"""
```

**What you should see / verify:**
Save the file, go back to the Streamlit app, and ask a question. The AI will immediately adopt your new, highly skeptical persona.

**Common problems & fixes:**

- _Problem:_ The prompt errors out after editing.
  _Fix:_ Ensure you did not accidentally delete the literal `{context}`, `{chat_history}`, or `{question}` placeholder variables inside the string. Expected syntax must remain intact.

**Next-checkpoint:** Proving your system works.

---

### Step 8: Basic Evaluation (Does it actually work?)

**Goal:** Run automated tests to prove the RAG pipeline returns the correct information without hallucinations.  
**Why it matters:** "Vibes" aren't a good testing strategy. An evaluation framework lets you know immediately if changing the chunk size or prompt broke the system's accuracy.

**Commands:**

```bash
python scripts/evaluate_rag.py --retrieval-only
```

**What you should see / verify:**
A table printed in the terminal showing 12 curated test questions (from `tests/golden_qa.json`) and what percentage of the right context the BM25/Vector search actually grabbed.

**Common problems & fixes:**

- _Problem:_ Missing `tests/golden_qa.json` errors.
  _Fix:_ Ensure you are running the script from the absolute root directory of the project, not from inside the `scripts/` folder.

**Next-checkpoint:** Expanding the knowledge base.

---

### Step 9: Adding New Documents on the Fly

**Goal:** Add more transcripts without tearing down the whole system.  
**Why it matters:** Knowledge bases grow over time. Rebuilding massive databases entirely from scratch is slow and expensive.

**Commands / action:**

1. Drop a new `.txt` or `.pdf` file into `data/transcripts/` using your file explorer, or use the **Upload Transcripts** button right in the Streamlit Sidebar.
2. Click the **🔄 Re-index** button in the Streamlit sidebar.

**What you should see / verify:**
The application will calculate file hashes, realize a transparent update is needed, process only the _new_ file, add it to ChromaDB, and refresh the UI in a few seconds.

**Common problems & fixes:**

- _Problem:_ The AI doesn't know about the newly uploaded file.
  _Fix:_ You likely forgot to click the **Re-index** button. The app does not watch the folder live; you must trigger the update.

---

_(Optional: Production Paths)_

### Optional: Running 100% Locally with Ollama

If you want absolute privacy (zero data sent to the cloud):

1. Install [Ollama](https://ollama.com).
2. Run `ollama serve` in a new terminal window.
3. Run `ollama pull gemma3:1b` (a fast, lightweight local model).
4. In the Streamlit Sidebar, switch the LLM Provider to **Ollama (Local & Private)**. Your queries now run entirely on your own hardware!

---

## ❓ FAQ / Troubleshooting

**Q: Why do I need "Hybrid search"? Isn't Vector AI magic?**  
A: Vectors understand meaning (e.g., "puppy" is close to "dog"). They are terrible at exact serial numbers, acronyms (like "VCP"), or specific trader names. BM25 (keyword search) solves the exact-match problem. Hybrid combines the best of both.

**Q: How do I reduce "hallucination"?**  
A: Lower the temperature in the sidebar to `0.0`, ensure the prompt template strictly demands "If you don't know, say 'Not in my notes'", and rely on the Cross-Encoder re-ranking to filter out weakly-associated context.

**Q: How do I update the index if I edit an existing `.txt` file?**  
A: The script `src/build_index.py` uses MD5 hashing! Just click the `Re-index` button in the UI. It will automatically detect the edits, delete the old chunks, and add the new ones.

**Q: Can I use a larger chunk size for better context?**  
A: Yes. Edit `chunk_size` inside `src/build_index.py`. However, bigger chunks mean you retrieve fewer diverse ideas, and you risk overflowing the LLM's context window.

**Q: The Streamlit UI says "Vector database not loaded"?**  
A: You likely skipped indexing. Stop the server, run `python src/build_index.py`, verify `data/chroma_db` is created, and restart Streamlit.

---

## 🔒 Security & Cost Notes

- **API Keys:** Never commit `.env` or hardcode `GROQ_API_KEY`. It gives access to your billing account.
- **Embedding Cost:** Completely free! We use HuggingFace embeddings running locally on your CPU instead of paying for OpenAI's `text-embedding-3`.
- **Inference Cost:** Groq has a very generous free tier (at the time of writing). If you switch to Ollama, inference is 100% free and offline.
- **Data Privacy (PII):** If you use Ollama, none of your documents ever leave your computer. If you use Groq, your searched chunks _are_ sent to Groq's API. Do not upload sensitive passwords or highly confidential personal data if routing to cloud LLMs.

---

## 🚀 What to build next

Make your resume shine by implementing one of these features to the existing codebase! All of these can easily hook into the current architecture:

1. **Add Real-Time Stock Data:** Extend `src/app.py` by integrating the `yfinance` Python library. If the user asks for a ticker, pull the live price and inject it directly into the Prompt Context before querying the LLM.
2. **Add Streaming UI Responses:** Though Groq is fast, make it feel instant. Look up LangChain's `astream` and Streamlit's `st.write_stream()` to render the text as it generates, matching modern ChatGPT aesthetics.
3. **Dockerize the Pipeline:** Add a simple `Dockerfile` using `python:3.11-slim` that copies your requirements, installs dependencies, and exposes port 8501. This shows you understand dev-ops and deployment.
4. **Export Chat History:** You are already saving chat turns to SQLite (`chat_history.db`). Add a button in the sidebar that executes a `SELECT *` query and allows the user to download their chat history as a `.csv` file.

Happy Building!
