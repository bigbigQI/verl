#!/usr/bin/env bash
# Apply the MXFP8 rollout patches directly to the sglang package that Python
# actually imports at runtime, in place. This is more robust than the
# PYTHONPATH-injection approach (prepare_sglang_mxfp8.sh): it patches the one
# directory every process resolves `import sglang` to — driver, Ray actors,
# and the SGLang HTTP server subprocess alike — so there is no environment to
# propagate and nothing to get out of sync.
#
# Idempotent: re-running is a no-op once the endpoint is present.
#
# Usage:
#   bash scripts/apply_sglang_mxfp8_inplace.sh [PATCH_DIR]
# PATCH_DIR defaults to the directory containing this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${1:-${SCRIPT_DIR}}"
POST_PROCESS_PATCH="${PATCH_DIR}/sglang_post_process_weights_24657.patch"
RMSNORM_PATCH="${PATCH_DIR}/sglang_force_native_rmsnorm.patch"

for f in "${POST_PROCESS_PATCH}" "${RMSNORM_PATCH}"; do
    [ -f "$f" ] || { echo "ERROR: patch not found: $f" >&2; exit 1; }
done

# Resolve the sglang package dir that THIS python actually imports.
SGLANG_PKG="$(python3 -c 'import sglang, os; print(os.path.realpath(os.path.dirname(sglang.__file__)))')"
# The patches reference paths as `python/sglang/srt/...`; with -p2 the leading
# `python/` is stripped to `sglang/srt/...`, so apply from the package parent.
SGLANG_ROOT="$(dirname "${SGLANG_PKG}")"
echo "sglang package : ${SGLANG_PKG}" >&2
echo "patch root (-p2): ${SGLANG_ROOT}" >&2

# Already patched? The post_process endpoint is the marker.
if python3 -c 'from sglang.srt.managers.io_struct import PostProcessWeightsReqInput' 2>/dev/null; then
    echo "Already patched: PostProcessWeightsReqInput present. Nothing to do." >&2
    exit 0
fi

apply_patch() {
    local patch_file="$1"
    # Skip patches whose changes are already fully present.
    if patch -d "${SGLANG_ROOT}" -p2 --forward --dry-run -R < "${patch_file}" >/dev/null 2>&1; then
        echo "Skip (already applied): $(basename "${patch_file}")" >&2
        return 0
    fi
    # Verify it applies cleanly BEFORE touching any file.
    if ! patch -d "${SGLANG_ROOT}" -p2 --forward --dry-run < "${patch_file}" >/dev/null 2>&1; then
        echo "ERROR: ${patch_file} does not apply cleanly to ${SGLANG_PKG}." >&2
        echo "       The installed sglang version likely diverged from the patch base" >&2
        echo "       (df38ef5 / PR #24657). Inspect with:" >&2
        echo "         patch -d ${SGLANG_ROOT} -p2 --dry-run < ${patch_file}" >&2
        exit 1
    fi
    # Apply, keeping .orig backups so the patch can be reverted.
    patch -d "${SGLANG_ROOT}" -p2 --forward -b < "${patch_file}" >&2
    echo "Applied: $(basename "${patch_file}")" >&2
}

apply_patch "${POST_PROCESS_PATCH}"
apply_patch "${RMSNORM_PATCH}"

# Verify the endpoint type now imports from the real package.
python3 -c 'from sglang.srt.managers.io_struct import PostProcessWeightsReqInput' \
    && echo "OK: /post_process_weights support is now installed in ${SGLANG_PKG}" >&2
