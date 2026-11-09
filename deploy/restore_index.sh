#!/bin/sh
# Restore the Colab ColPali/Qdrant index into the compose Qdrant server, then hand the files to the app user.
#
#   docker compose run --rm --user root api sh deploy/restore_index.sh
#
# Reads the snapshots and ingest.sqlite3 from /snapshots (SNAPSHOT_DIR in .env, copied from
# MyDrive/capstone_rag/snapshots/colpali_qdrant). /qdrant/snapshots is a volume shared with the
# Qdrant container at the same path, so the server can read the file it is told to recover.
set -eu
ROOT=/app/indexes/financial_qdrant
mkdir -p "$ROOT"
python -m scripts.rag.financial_ingest restore --root "$ROOT" --qdrant-url "${QDRANT_URL:-http://qdrant:6333}" \
    --snapshot-dir /snapshots --server-snapshots-path /qdrant/snapshots
chown -R "${APP_UID:-1000}:${APP_GID:-1000}" /app/indexes
echo "restored into $ROOT; restart the API: docker compose restart api"
