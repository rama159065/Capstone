# Embedding API Key: Setup and Usage

How to use the Track B shared embedding API (`bge-small-en-v1.5`) from this project: where the key and certificate go, how to build and serve an index with it, its limits, and how to fix common errors. Date: 14 November 2026.

**This file never contains the key.** The key comes from the course coordinator (`shared-embedding-key.txt` in the `VM02_Embeddings_Access` folder) and goes only into your local `.env` or a secret store.

**In short**

- **Service:** `POST https://103.42.50.41:1035/v1/embeddings`, model `bge-small-en-v1.5`, OpenAI-compatible request and response, 384 numbers per vector (already normalised).
- **You need three things:** the key (`EMBEDDING_API_KEY`), the private CA certificate (`vm02-ca.crt`, saved as `RAG_HOME/secrets/embedding_ca.crt`) and three settings in `.env`.
- **Limits:** at most **16 texts** per request and **512 tokens** per text. The project's client keeps to both by itself.
- **Same vectors as local `BAAI/bge-small-en-v1.5`** (cosine 1.0 on every text checked), so pipeline 1 treats them as one model: an index built with either is continued and searched with either.

---

## 1. The files from the coordinator

| File | What it is | Where it goes |
|---|---|---|
| `Embedding_API_Quickstart.md` | The service's address, model, limits and example `curl` calls | Reference only |
| `shared-embedding-key.txt` | The API key: one line starting `sk-`, sent as `Authorization: Bearer <key>` | Your `.env` (`EMBEDDING_API_KEY`), never into git |
| `vm02-ca.crt` | Root certificate of the server's private certificate authority (Caddy Local Authority, 2026 ECC root, valid to 2036). The server's HTTPS certificate is signed by it, so normal certificate checks reject the server without it | `indexes/rag_home/secrets/embedding_ca.crt` (gitignored) |

## 2. Set up (once per machine)

**Step 1: the certificate.** Copy it into the secrets folder of your `RAG_HOME` (local default `indexes/rag_home`):

```powershell
New-Item -ItemType Directory -Force indexes\rag_home\secrets | Out-Null
Copy-Item C:\Upgrad_Projects\IIT_Kharagpur\VM02_Embeddings_Access\vm02-ca.crt indexes\rag_home\secrets\embedding_ca.crt
```

The runtime profile sets `EMBEDDING_CA_CERT` to this file when it exists. To keep the certificate elsewhere, set `EMBEDDING_CA_CERT` to its path. Don't install it in the Windows certificate store: it should be trusted for this service only.

**Step 2: the key.** Add it to `.env` in the repo root (`.env` is gitignored). The address and model default to the shared API, so the key is the only setting you must add:

```
EMBEDDING_API_KEY=<paste the key from shared-embedding-key.txt>
# optional, these are the defaults:
EMBEDDING_API_BASE=https://103.42.50.41:1035/v1
EMBEDDING_MODEL=bge-small-en-v1.5
EMBEDDING_MAX_BATCH=16
EMBEDDING_MAX_TOKENS=512
# which one pipeline 1 uses to embed questions: local or hosted (server and gcp profiles: hosted)
RAG_EMBEDDING_PROVIDER=
```

**Step 3: check it.** One request with the quickstart's PowerShell form (it prints a JSON reply with two vectors of 384 numbers):

```powershell
$key = (Get-Content C:\Upgrad_Projects\IIT_Kharagpur\VM02_Embeddings_Access\shared-embedding-key.txt -Raw).Trim()
$body = @{ model = 'bge-small-en-v1.5'; input = @('Example document chunk', 'Example search query') } | ConvertTo-Json -Compress
$body | curl.exe --cacert indexes\rag_home\secrets\embedding_ca.crt https://103.42.50.41:1035/v1/embeddings `
  -H "Authorization: Bearer $key" -H "Content-Type: application/json" --data-binary '@-'
```

## 3. Use it in the project

### Build an index

Pipeline 1 (`financial_ingest_chroma_db`) uses bge-small-en-v1.5 by default: **locally** (`BAAI/bge-small-en-v1.5`, no flag needed) or **through the API** (`--embedding-provider hosted`, or `RAG_EMBEDDING_PROVIDER=hosted`). Both give the same vectors, so one index can be built partly on this PC, partly through the API or on Colab, without redoing finished filings. Use a **new index folder**: the current index (`indexes/rag_home/indexes/chroma_db`) was built with `all-MiniLM-L6-v2` and refuses bge-small.

```powershell
# Pipeline 1 (Chroma), one sector as a start; reuses the OCR cache, so pages are not read again
python -m scripts.rag.financial_ingest_chroma_db ingest `
  --root indexes\rag_home\indexes\chroma_db_bge `
  --cache-dir indexes\rag_home\cache\vlm_pages `
  --manifest data\raw\financial_report_qa\annual_reports --corpus-preset min500 --sectors healthcare `
  --vlm routed --skip-routes qwen3VL,paidVL --no-describe-visuals --workers 6 --low-priority `
  --embedding-provider hosted
```

The index records the first builder (`local:BAAI/bge-small-en-v1.5:384` or `hosted:bge-small-en-v1.5:384`); either is accepted afterwards. Pipeline 2 (`financial_ingest_qdrant_db`) also takes `--embedding-provider hosted`, but still defaults to `all-MiniLM-L6-v2`.

### Serve and search

Point the API at the new folder and start it as usual. For each question it embeds with `RAG_EMBEDDING_PROVIDER` (unset: the provider that built the index; the `server` and `gcp` profiles set `hosted`). If that provider can't start (no key, API unreachable, model not cached), it uses the other one and logs `embedding provider fallback`; results are the same either way (checked: identical top 5 for every test question).

```powershell
$env:RAG_VLM_DIR = "indexes\rag_home\indexes\chroma_db_bge"
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
```

### What the client does for you

| Limit or need | Handled by `Embedder` in `scripts/rag/financial_ingest_qdrant_db.py` |
|---|---|
| Private certificate | Every request trusts `EMBEDDING_CA_CERT`; a missing file is an error at start-up |
| 16 texts per request | Texts are sent in batches of `EMBEDDING_MAX_BATCH`; a larger batch is refused before sending |
| 512 tokens per text | Each text is cut at a token boundary to `EMBEDDING_MAX_TOKENS`, counted with the model's own tokenizer (`BAAI/bge-small-en-v1.5`, about 134 MB, from the Hugging Face cache). Only the vector is made from the cut text; the full chunk is still stored, shown and searched by BM25. Without the tokenizer it cuts at `EMBEDDING_FALLBACK_CHARS` (1,200) characters |
| Overload | Retries after 2, 4 and 8 seconds on HTTP 429 or 503 |
| Wrong reply | Fails if the number of vectors differs from the number of texts |

About 10% of the current chunks are longer than 512 tokens (mostly long statement tables) and are cut for their vector.

## 4. Check results (14 November 2026)

| Check | Result |
|---|---|
| Two texts | 200 OK, 384 numbers each, length 1.0 |
| Without the certificate | Refused by the client: `CERTIFICATE_VERIFY_FAILED` |
| Wrong key | 401: "A valid key for this endpoint is required." |
| 17 texts in one request | 422: "batch size 17 > maximum allowed batch size 16" |
| Token limit | 512 tokens accepted, 513 refused with 422 (the message says "less than 512", one short of what it accepts) |
| Service against local `BAAI/bge-small-en-v1.5` | Identical vectors (cosine 1.0 on 16 real chunks and a query), identical ranking |
| Speed | About 0.5 s per request of 16 chunks, so about 1.3 hours for 132,600 chunks |
| End to end | AbbVie FY2025: 807 chunks embedded (17 cut to 512 tokens); searches returned the right pages (Humira revenue, R&D expense) in about 130 to 150 ms each |

## 5. Errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `CERTIFICATE_VERIFY_FAILED` | The certificate isn't used | Put it at `RAG_HOME/secrets/embedding_ca.crt` or set `EMBEDDING_CA_CERT` |
| `EMBEDDING_CA_CERT not found: ...` | The path is wrong | Fix the path, or unset the variable to use the default |
| `Missing hosted embedding env var(s): ...` | `.env` not loaded or a setting missing | Set `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` |
| HTTP 401 | Wrong, expired or rotated key | Get the current key from the coordinator |
| HTTP 422 batch size | A client that sends more than 16 texts | Use this project's client, or `EMBEDDING_MAX_BATCH=16` |
| HTTP 422 tokens | A text over 512 tokens | Use this project's client; keep `EMBEDDING_MAX_TOKENS` at 512 or lower |
| HTTP 429 or 503 | The shared server is busy | Retried automatically; for long builds use fewer workers or run at quieter times |
| `embedding differs from the existing index` | The folder was built with another model (for example MiniLM) | Build into a new folder (`--root`), or pass that index's model; local and hosted bge-small are never refused against each other |
| Timeouts | Network or the server is down | Check with the `curl.exe` call in section 2; the API's searches fail too while the service is down |

## 6. Hosted service or local model?

The service and the local model give the same vectors; the difference is where the work runs.

| | Hosted service | Local `BAAI/bge-small-en-v1.5` |
|---|---|---|
| Building 100,000+ chunks | About 1 to 1.5 hours, over the network, on a server other students share | About 1 to 2 hours on this PC's CPU, minutes on a Colab GPU (estimate); no network |
| Each question | One network call (about 0.1 s) | Local; no outside dependency |
| Needs | Key, certificate, reachable server | The model in the Hugging Face cache (134 MB, already on this PC) |
| Index records | `hosted:bge-small-en-v1.5:384` | `local:BAAI/bge-small-en-v1.5:384` (the two are interchangeable in pipeline 1) |

For pipeline 1 you don't have to choose per index: build with whichever is faster at the time (local or Colab GPU for big builds, the API when this PC is busy) and serve with either. Only bge-small-en-v1.5 is treated this way. The current index (`indexes/rag_home/indexes/chroma_db`) was built with `sentence-transformers/all-MiniLM-L6-v2`; it keeps using MiniLM, and resuming it needs `--embedding-model sentence-transformers/all-MiniLM-L6-v2`.

## 7. Colab and Google Cloud

The certificate and key are never copied into git, Drive or a bucket (`common/storage.py`).

| Where | Key | Certificate |
|---|---|---|
| Colab | Colab's **Secrets** panel (`EMBEDDING_API_KEY`), read with `google.colab.userdata.get` into `os.environ` | Upload `vm02-ca.crt` in the session to `/content/rag/secrets/embedding_ca.crt` |
| Docker or a GCP VM | `.env` beside `docker-compose.yml`, readable only by you (`chmod 600 .env`) | `./secrets/embedding_ca.crt` beside `docker-compose.yml` (or the folder `SECRETS_DIR` names); compose mounts it read-only at `/app/rag_home/secrets` |

The server must accept connections from where you run: test with the `curl` call first, because a cloud VM's address may not be allowed.

## 8. Handling the key

- **Never commit it.** Not in code, notebooks, docs, logs or screenshots. `.env`, `secrets/` and `*.crt` are gitignored; check `git status` before every commit.
- **It is shared.** Everyone on the course uses the same key, so it identifies the course, not you. Don't post it anywhere others outside the course can read, including shared Drive folders and public repositories.
- **If it leaks,** tell the coordinator so it can be rotated, and remove it from where it was posted (git history included).
- **Load it, don't type it.** Read it from the file or `.env` (as in section 2); don't paste it into command lines that shell history keeps.
- **Go easy on the shared server.** Building the whole corpus is about 8,000 or more requests; run long builds once, from the OCR cache, and not in parallel with other large jobs.
