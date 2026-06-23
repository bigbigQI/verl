#!/usr/bin/env bash
set -euo pipefail

patched_dir="${1:?Usage: prepare_sglang_mxfp8.sh PATCHED_SGLANG_DIR}"
patch_file="${2:-/apps/verl/scripts/sglang_post_process_weights_24657.patch}"
native_rmsnorm_patch="${3:-/apps/verl/scripts/sglang_force_native_rmsnorm.patch}"

if [ ! -d "${patched_dir}/python/sglang" ]; then
    cp -a /sgl-workspace/sglang "${patched_dir}"
    patch -d "${patched_dir}" -p1 < "${patch_file}" >&2
    patch -d "${patched_dir}" -p1 < "${native_rmsnorm_patch}" >&2
fi

PYTHONPATH="${patched_dir}/python:${PYTHONPATH:-}" \
    python3 -c 'from sglang.srt.managers.io_struct import PostProcessWeightsReqInput'

printf '%s\n' "${patched_dir}/python"
