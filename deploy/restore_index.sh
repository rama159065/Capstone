#!/bin/sh
# Put both indexes in place for the compose stack, then hand the files to the app user.
#
#   docker compose run --rm --user root api sh deploy/restore_index.sh
#
# Pipeline 2 (Qdrant): restores the snapshots into the compose Qdrant server and the ingest state into
#   <RAG_HOME>/indexes/qdrant_db. Snapshots are read from <RAG_HOME>/snapshots/qdrant_db (deploy/sync_from_gcs.sh
#   puts them there), else from /snapshots/qdrant_db or /snapshots (SNAPSHOT_DIR in .env, copied from
#   MyDrive/capstone_rag/snapshots/qdrant_db or the older snapshots/colpali_qdrant).
#   /qdrant/snapshots is a volume shared with the Qdrant container at the same path, so the server can read the
#   file it is told to recover.
# Pipeline 1 (Chroma): a plain folder copy. Already in place when <RAG_HOME>/indexes/chroma_db/vlm_ingest.sqlite3
#   exists (sync_from_gcs.sh); otherwise copied from /snapshots/chroma_db (or the older /snapshots/vlm_chroma).
set -eu
HOME_DIR="${RAG_HOME:-/app/rag_home}"
QDRANT_ROOT="$HOME_DIR/indexes/qdrant_db"
CHROMA_ROOT="$HOME_DIR/indexes/chroma_db"
has_snapshots() { ls "$1"/*.snapshot >/dev/null 2>&1; }

SNAPS=""
for dir in "$HOME_DIR/snapshots/qdrant_db" /snapshots/qdrant_db /snapshots/colpali_qdrant /snapshots; do
    if has_snapshots "$dir"; then SNAPS="$dir"; break; fi
done
if [ -n "$SNAPS" ]; then
    mkdir -p "$QDRANT_ROOT"
    python -m scripts.rag.financial_ingest_qdrant_db restore --root "$QDRANT_ROOT" \
        --qdrant-url "${QDRANT_URL:-http://qdrant:6333}" --snapshot-dir "$SNAPS" --server-snapshots-path /qdrant/snapshots
    echo "qdrant: restored from $SNAPS into $QDRANT_ROOT"
else
    echo "qdrant: no snapshots found (pipeline 2 skipped)"
fi

if [ -f "$CHROMA_ROOT/vlm_ingest.sqlite3" ]; then
    echo "chroma: already in place at $CHROMA_ROOT"
else
    for dir in /snapshots/chroma_db /snapshots/vlm_chroma; do
        if [ -f "$dir/vlm_ingest.sqlite3" ]; then
            mkdir -p "$CHROMA_ROOT"
            cp -R "$dir/." "$CHROMA_ROOT/"
            echo "chroma: copied $dir into $CHROMA_ROOT"
            break
        fi
    done
    [ -f "$CHROMA_ROOT/vlm_ingest.sqlite3" ] || echo "chroma: no index found (pipeline 1 skipped)"
fi

mkdir -p "$HOME_DIR/state"
for dir in "$HOME_DIR/indexes" "$HOME_DIR/snapshots" "$HOME_DIR/state" /app/indexes; do
    [ -d "$dir" ] && chown -R "${APP_UID:-1000}:${APP_GID:-1000}" "$dir"
done
echo "done; restart the API: docker compose restart api"
