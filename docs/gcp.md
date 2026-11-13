# Deploying the Annual Report Copilot on Google Cloud

Written 13 November 2026. This page covers how long a Google Cloud deployment takes, which components and services it needs, and what it costs to run for 15 days. The settings, paths and scripts behind it are in [deployment.md](deployment.md) §7 (storage layout and the `gcp` profile).

## Summary

| | Full corpus (1,004 filings) | `min500` (509 filings) |
|---|---|---|
| Hands-on setup on Google Cloud (index already built) | ~2–3 h the first time, ~30 min for redeploys | same |
| End to end, including the one-off index build on Colab | 1–2 days | 1–2 days (most of it Qwen3-VL GPU time) |
| Google Cloud cost for 15 days | **≈ $55** | **≈ $29** |
| Plus the one-off index build (Colab Pro + Drive) | + $12 → **≈ $67** | + $12 → **≈ $41** |
| LLM | Track B quota | Track B quota |

**Where the prices come from:**
- VM and storage prices are third-party list prices for `us-central1`, taken from [deployment.md](deployment.md) §5–6.
- The static IP charge was added here, and everything was scaled from 14 to 15 days.
- Check the Google Cloud pricing calculator for your region before creating anything.

```
Colab: build into /content/rag ──after each sector──► Drive (MyDrive/capstone_rag) ──cell 12 / cp.export_to_gcs──► gs://bucket
GCP VM: sh deploy/sync_from_gcs.sh  (bucket ──gcloud storage rsync──► RAG_HOME_DIR, Qdrant restore, docker compose up)
```

## 1. Time to deploy

### 1.1 Build the index (one-off, on Colab)

This is the slow part.

| Step | min500 (509 filings) | Full (1,004) |
|---|---|---|
| Download the PDFs from EDGAR to Drive | ~15–30 min | ~30–60 min |
| Page audit (optional, CPU) | ~3–4 h | ~6–8 h |
| Pipeline 1 (Chroma): text layer + OCR | ~2.5–3.5 h | ~5–8 h |
| Pipeline 1: Qwen3-VL pages + chart descriptions (~2.5k pages) | ~10–28 GPU-hours on a T4 (less on an L4) | about 2× |
| Pipeline 2 (Qdrant): text + ColPali (only if you deploy it) | ~4–7 h | ~8–15 h |

These times are estimates from measurements on a few filings. The first full sector gives the real numbers (`cp.status`).

### 1.2 Set up on Google Cloud

| # | Step | Time |
|---|---|---|
| 1 | Create a project, attach billing, create the bucket | ~15 min |
| 2 | Copy Drive → bucket: cell 12 of either ingest notebook, or `cp.export_to_gcs` | ~10–20 min |
| 3 | Create the VM, persistent disk, static IP and firewall rules (22, 80, 443) | ~15 min |
| 4 | Install Docker; clone the repo; write `.env`; put the CA certificate in `secrets/` | ~20 min |
| 5 | First `docker compose build` (Python + CPU torch) | ~10–15 min |
| 6 | `sh deploy/sync_from_gcs.sh`: pull the indexes, restore Qdrant, start the stack | ~10–20 min |
| 7 | HTTPS via Caddy + `sslip.io`, then smoke tests (health and the three agents) | ~20–30 min |
| | **Total, first time** | **~2–3 h**; later redeploys ~30 min |

## 2. Components and services

### 2.1 Google Cloud

| Service | What for | Setting |
|---|---|---|
| Compute Engine VM | Runs the stack | `e2-standard-4` (4 vCPU / 16 GB) for full; `e2-standard-2` (2 vCPU / 8 GB) for min500; Ubuntu 24.04 LTS |
| Persistent disk (balanced SSD) | `RAG_HOME_DIR` (indexes, state), Docker images, model cache | 100 GB full / 50 GB min500 |
| Static external IP | A stable address for HTTPS | 1 IPv4 |
| VPC firewall rules | Only 22 (SSH, your IP), 80 and 443 are open; 8000, 8501 and 6333 stay closed | — |
| Cloud Storage bucket | Index copies from Colab: `indexes/chroma_db` and `snapshots/qdrant_db` | ~5–20 GB, same region as the VM |
| Default service account | Lets the VM read the bucket, with no keys on the server | Storage read access |

### 2.2 On the VM (Docker Compose)

One image serves both `api` and `ui`.

| Container | Role | Port |
|---|---|---|
| `caddy` (`https` profile) | HTTPS (Let's Encrypt) and a basic-auth login in front of the UI | 80, 443 (public) |
| `ui` | Streamlit | 8501 (localhost only) |
| `api` | FastAPI with the three agents (router, Q&A, trend). MiniLM and the reranker run on CPU here, and Pipeline 1's Chroma index is read from disk | 8000 (localhost only) |
| `qdrant` | Pipeline 2 collections (`annual_reports_text`, `annual_reports_visual`) | 6333 (internal) |

### 2.3 Outside Google Cloud

| Service | Role | Cost |
|---|---|---|
| Track B LLM endpoint (Qwen3-8B) | All answers and the critic's fact-checking judge | team quota |
| Colab Pro + Google Drive 100 GB | Index build, and its long-term copy | $9.99 + $1.99 per month |
| GitHub | The code the VM clones | free |
| LangSmith (optional) | Traces | free tier |
| `sslip.io` hostname | HTTPS without buying a domain (own domain ~$10–15/year) | free |
| Hugging Face Hub | Model downloads on first start, then cached | free |

## 3. Cost for 15 days

| Item | Full corpus | min500 |
|---|---|---|
| VM compute (24 h × 15 days = 360 h) | e2-standard-4: 0.134 × 360 ≈ **$48** | e2-standard-2: 0.067 × 360 ≈ **$24** |
| Persistent disk | 100 GB ≈ **$5** | 50 GB ≈ **$2.5** |
| Static IP (in use) | ≈ $0.005/h ≈ **$2** | ≈ **$2** |
| Cloud Storage bucket | 15–20 GB ≈ **$0.20** | ~8 GB ≈ **$0.10** |
| Network egress (a demo's traffic) | < **$1** | < **$1** |
| **Google Cloud for 15 days** | **≈ $55** | **≈ $29** |
| One-off index build: Colab Pro + Drive | $12 | $12 |
| LLM | Track B quota | Track B quota |
| **Total** | **≈ $67** | **≈ $41** |

**Optional:** a paid vision model for Pipeline 1's `paidVL` pages adds about **$1–$39** for min500, depending on the model. This is pending the mentor's decision ([chart_reading.md](chart_reading.md)).

### 3.1 Paying less

- **Run the VM only when needed.** A stopped VM costs no compute, but the disk and the IP are still charged. Running 8 hours a day cuts the compute line to about a third.
- **Use the free trial.** New Google Cloud accounts get free trial credit, which would likely cover all of this; check the current terms when you sign up.
- **Choose min500.** It halves the VM size and the disk.
- **Delete everything after grading:** the VM **and** its disk, the static IP and the bucket. Otherwise billing continues.

## 4. Step-by-step commands

**On Colab, after ingestion:** run cell 12 of `colab_ingest_chroma_db.ipynb` or `colab_ingest_qdrant_db.ipynb`, or:

```bash
python -m scripts.rag.colab_pipeline export --pipeline vlm_chroma --bucket gs://capstone-rag-<name>
```

This authenticates, creates the bucket if needed, and copies both indexes from Drive.

**On the VM:**

```bash
# once: Docker Engine + Compose plugin, then the code
git clone <repo-url> ~/capstone && cd ~/capstone
cp .env.example .env && chmod 600 .env
#   RAG_PROFILE=gcp
#   RAG_GCS_BUCKET=gs://capstone-rag-<name>
#   RAG_HOME_DIR=/srv/capstone          (on the persistent disk)
#   LLM_API_BASE / LLM_API_KEY / LLM_MODEL, API_AUTH_TOKEN, DOMAIN=<ip>.sslip.io, BASIC_AUTH_HASH
mkdir -p secrets && cp <path>/llm_ca.crt secrets/llm_ca.crt
sudo mkdir -p /srv/capstone && sudo chown -R $USER /srv/capstone

# pull both indexes from the bucket, restore Qdrant, start the stack
sh deploy/sync_from_gcs.sh
docker compose --profile https up -d

# check
curl -fsS http://127.0.0.1:8000/health      # profile gcp, both index folders, reranker / ca_cert flags
```

**After grading:**

```bash
docker compose down
# then delete the VM, its disk, the static IP and the bucket in the console (or with gcloud)
```

## 5. Not verified yet

- **Docker image:** not built yet, because Docker isn't installed on the development PC. The first build on the VM is the real check.
- **Index copy scripts:** the `gcloud` steps (`cp.export_to_gcs`, `deploy/sync_from_gcs.sh`) haven't been run against a real bucket. Their syntax and the commands they build are tested.
- **Qwen3-VL and ColPali at full scale:** neither has run on a Colab GPU yet, so the GPU-hour estimates are uncertain.
- **Prices:** these are list prices. Some LLM and Colab prices change on 1 December and 1 January.

**Do a dry run first.** Before the full sync, deploy 1–2 sectors end to end (about an hour) to confirm the build, the sync and the restore.
