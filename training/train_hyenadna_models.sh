#!/usr/bin/env bash
set -euo pipefail

# Train the same A/C/G/T projection head across the official HyenaDNA model sizes.
# Models run sequentially so only one frozen backbone occupies memory at a time.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname -- "${script_dir}")"
cd "${repo_root}"

models=(
  "LongSafari/hyenadna-tiny-1k-seqlen-hf"
  "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf"
  "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
  "LongSafari/hyenadna-small-32k-seqlen-hf"
  "LongSafari/hyenadna-medium-160k-seqlen-hf"
  "LongSafari/hyenadna-medium-450k-seqlen-hf"
  "LongSafari/hyenadna-large-1m-seqlen-hf"
)

mkdir -p training/heads training/runs

for model in "${models[@]}"; do
  model_id="${model##*/}"
  model_data="${model_id#hyenadna-}"
  model_data="${model_data/-seqlen-/-}"
  model_data="${model_data%-hf}"
  log_path="training/runs/train_hyenadna_${model_data}.log"
  metrics_path="training/runs/train_hyenadna_${model_data}.metrics.json"

  echo "Training ${model}"
  echo "Logging to ${log_path}"
  uv run --frozen python -m training.train_projection_head \
    --model-name "${model}" \
    --log-file "${log_path}" \
    --metrics-file "${metrics_path}" \
    --fasta-files 'training/data/circadian/*.fasta' \
    --fasta-test-files 'training/data/circadian-held-out/*.fasta' \
    --fasta-test-files 'training/data/non-circadian/*.fasta' \
    --stride 16 \
    --device auto
done
