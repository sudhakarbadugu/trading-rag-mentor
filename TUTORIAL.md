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
