#!/bin/bash
# Populate the transformers_modules cache with jina custom code for offline use.
# Run this once after setup, or again whenever the modules cache is cleared.
set -e

# ==== EDIT THESE PATHS to where the jina folders actually live ====
JINA_CLIP_V2_DIR="${JINA_CLIP_V2_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/jina-clip-v2" && pwd)}"
JINA_EMB_V3_DIR="${JINA_EMB_V3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/jina-embeddings-v3" && pwd)}"

TM_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/modules/transformers_modules"

copy_py() {
    local src_dir="$1"
    local name
    name="$(basename "$src_dir")"
    local dst_dir="$TM_CACHE/$name"

    if [ ! -d "$src_dir" ]; then
        echo "[SKIP] source not found: $src_dir"
        return
    fi
    mkdir -p "$dst_dir"
    cp "$src_dir"/*.py "$dst_dir"/
    echo "[OK] $name : $(ls "$dst_dir"/*.py | wc -l) .py files -> $dst_dir"
}

copy_py "$JINA_CLIP_V2_DIR"
copy_py "$JINA_EMB_V3_DIR"
echo "[DONE] offline HF custom modules populated"
