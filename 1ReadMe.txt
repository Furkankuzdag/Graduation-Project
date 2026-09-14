# RAG System — Setup & Run Instructions

## Project Structure

Note: I initially developed each model separately, but later merged them into app2.py. The models currently support English vocabulary only.
This project now runs as a **single Streamlit application** (`app2.py`).
There is no need for separate `app_word2vec.py`, `app_fasttext.py`, or `app_glove.py` files —
the embedding model and the chunking method are both chosen from dropdowns/menus inside the app itself.

| Embedding Model | 
|------|
| GloVe | 
| Word2Vec | 
| FastText | 

The in-app flow is: **Upload PDF → choose chunking method (Semantic / Sliding Window) → choose embedding model (GloVe / Word2Vec / FastText) → Build Log**.

Every "Build Log" run writes which model and chunking method it used to `rag_meta.txt`. The "Ask Question" tab reads this file automatically and loads the matching model — there is no manual log-switching between models anymore.

## Required Files

Place all of the following files in the **same folder**:

- `app2.py`
- `glove.2024.dolma.300d.zip` (only needed if you plan to use the GloVe option)
- your source PDF (e.g. `Mohamed Elgendy - Deep Learning for Vision Systems MEAP V06 (2019, Manning Publications Co.) - libgen.li.pdf`)

`wiki_word2vec.model` and `fasttext_enwiki_20260401_cbow_epoch1.bin` are **no longer needed** —
Word2Vec and FastText are trained from scratch on whichever PDF you upload, so no pretrained files for them are required.

## Requirements

Install dependencies:

```bash
pip install streamlit PyPDF2 numpy gensim scikit-learn google-generativeai
```

## How to Run

Only one command is needed:

```bash
streamlit run app2.py
```

Then inside the app:

1. Go to the **Build Log** tab and upload your PDF.
2. Pick a chunking method: **Semantic Chunking** or **Sliding Window**.
3. Pick an embedding model: **GloVe**, **Word2Vec**, or **FastText**. If you pick Word2Vec/FastText you can adjust their hyperparameters in the sidebar (the defaults are the ones used in the thesis experiments).
4. Click **Build Log** — it builds the log and writes the matching metadata automatically.
5. Switch to the **Ask Question** tab and ask your question.

## Why a Single Log File Works Now

Previously each model needed its own log file because the vector dimensions differ (GloVe=300, Word2Vec=150, FastText=200), and mixing them produced broken or empty results.

Now there is one log file (`rag_log.txt`), but every "Build Log" run overwrites it and records which model/dimension it used in `rag_meta.txt`. The "Ask Question" tab reads that metadata first and loads the correct model automatically — so you just need to re-run **Build Log** whenever you switch model or chunking method, no manual file swapping required.

**Always build the log first** (click "Build Log" in the app) before asking questions.

## API Key

A free Google Gemini API key is required for answer generation.
Get one at: https://aistudio.google.com/app/apikey
