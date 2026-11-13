#!/bin/sh
# Google Cloud VM: pull both indexes from the bucket into RAG_HOME_DIR, restore Qdrant, and start the stack.
#
#   cd ~/capstone && sh deploy/sync_from_gcs.sh            # RAG_GCS_BUCKET and RAG_HOME_DIR come from .env
#
# The bucket holds the storage layout written from Colab by colab_pipeline.export_to_gcs (or `python -m
# scripts.rag.colab_pipeline export --bucket gs://...`):
#   gs://<bucket>/indexes/chroma_db/      pipeline 1: chroma/ + vlm_ingest.sqlite3, copied as is
#   gs://<bucket>/snapshots/qdrant_db/    pipeline 2: Qdrant snapshots + ingest.sqlite3, restored into the server
# The VM's default service account reads a bucket in the same project; no Google login on the server.
# Safe to rerun: rsync copies only what changed, and a Qdrant restore replaces the collections.
set -eu
cd "$(dirname "$0")/.."

env_value() {   # KEY from the environment, else from .env (without sourcing it: values may contain $ or spaces)
    eval "current=\${$1:-}"
    if [ -n "$current" ]; then printf '%s' "$current"; return; fi
    [ -f .env ] && sed -n "s/^$1=//p" .env | tail -n 1 | tr -d '\r'
}

BUCKET="$(env_value RAG_GCS_BUCKET)"
HOST_HOME="$(env_value RAG_HOME_DIR)"
HOST_HOME="${HOST_HOME:-./indexes/rag_home}"
if [ -z "$BUCKET" ]; then
    echo "set RAG_GCS_BUCKET (gs://name) in .env or the environment" >&2
    exit 1
fi
case "$BUCKET" in gs://*) ;; *) BUCKET="gs://$BUCKET" ;; esac
BUCKET="${BUCKET%/}"

mkdir -p "$HOST_HOME/indexes/chroma_db" "$HOST_HOME/snapshots/qdrant_db" "$HOST_HOME/state"
for pair in "indexes/chroma_db" "snapshots/qdrant_db"; do
    if gcloud storage ls "$BUCKET/$pair/" >/dev/null 2>&1; then
        gcloud storage rsync -r "$BUCKET/$pair" "$HOST_HOME/$pair"
        echo "synced $BUCKET/$pair -> $HOST_HOME/$pair"
    else
        echo "not in the bucket, skipped: $BUCKET/$pair"
    fi
done

docker compose up -d qdrant
docker compose run --rm --user root api sh deploy/restore_index.sh
docker compose up -d
echo "stack up: curl -fsS http://127.0.0.1:8000/health"
