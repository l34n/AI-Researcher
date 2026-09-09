#!/usr/bin/env bash
# L34N benchmark suite quickstart.
#
# Usage:
#   scripts/run_benchmark.sh                # free profile, full-fidelity
#   scripts/run_benchmark.sh local          # local Ollama models
#   scripts/run_benchmark.sh paid           # paid OpenRouter models
#   scripts/run_benchmark.sh all            # every model in the matrix
#   scripts/run_benchmark.sh paid --brief-only     # skip enrichment pass
#   scripts/run_benchmark.sh paid --matrix my.yaml # custom matrix
#
# Env:
#   OPENROUTER_API_KEY  required for free/paid tiers
#   OLLAMA_HOST         defaults to http://localhost:11434
#   L34N_ROOT           override repo root (defaults to script's parent dir)
#
# After the sweep completes, renders docs/benchmark-results.md from the
# per-model JSON files in data/benchmark-results/.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
export L34N_ROOT="${L34N_ROOT:-$HERE}"

profile="${1:-free}"
shift || true

case "$profile" in
  free|local|paid|all) ;;
  -h|--help)
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "unknown profile: $profile (expected: free | local | paid | all)" >&2
    exit 2
    ;;
esac

# Preflight
if [[ "$profile" == "free" || "$profile" == "paid" || "$profile" == "all" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is required for the '$profile' profile." >&2
    echo "Get a key at https://openrouter.ai/keys and export it:" >&2
    echo "  export OPENROUTER_API_KEY=sk-or-..." >&2
    exit 3
  fi
fi

if [[ "$profile" == "local" || "$profile" == "all" ]]; then
  host="${OLLAMA_HOST:-http://localhost:11434}"
  if ! curl -sf "$host/api/tags" >/dev/null 2>&1; then
    echo "Ollama not reachable at $host — start ollama or set OLLAMA_HOST." >&2
    [[ "$profile" == "local" ]] && exit 3
    echo "  (continuing; --profile all will still run free+paid tiers)" >&2
  fi
fi

py="${PYTHON:-python}"
"$py" -c "import ai_researcher" 2>/dev/null || {
  echo "ai_researcher package not importable from L34N_ROOT=$L34N_ROOT/src." >&2
  echo "Install the project first:  uv venv && uv pip install -e '.[dev]'" >&2
  exit 3
}

out_dir="$L34N_ROOT/data/benchmark-results/$profile"
mkdir -p "$out_dir"

echo "→ profile=$profile out=$out_dir"
"$py" "$HERE/scripts/benchmark_models.py" \
  --profile "$profile" --out "$out_dir" "$@"

# Render the aggregated report from every profile directory that exists.
report_out="$L34N_ROOT/docs/benchmark-results.md"
report_dirs=()
for d in free local paid; do
  [[ -d "$L34N_ROOT/data/benchmark-results/$d" ]] && report_dirs+=("$L34N_ROOT/data/benchmark-results/$d")
done

if [[ ${#report_dirs[@]} -gt 0 ]]; then
  echo "→ rendering report to $report_out"
  in_args=()
  for d in "${report_dirs[@]}"; do in_args+=(--in "$d"); done
  "$py" "$HERE/scripts/benchmark_report.py" --out "$report_out" "${in_args[@]}" || {
    echo "(report render failed; per-model JSON is still in $out_dir)" >&2
  }
fi

echo "done."
