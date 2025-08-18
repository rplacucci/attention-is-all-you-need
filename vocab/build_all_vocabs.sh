#!/usr/bin/env bash
set -euo pipefail

# Defaults
VOCAB_SIZE=37000
MIN_FREQUENCY=2

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vocab_size)
      [[ $# -ge 2 ]] || { echo "Error: --vocab_size needs value" >&2; exit 1; }
      VOCAB_SIZE="$2"
      shift 2
      ;;
    --min_frequency)
      [[ $# -ge 2 ]] || { echo "Error: --min_frequency needs value" >&2; exit 1; }
      MIN_FREQUENCY="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

LANGS=(cs de fr hi ru)

for LANG in "${LANGS[@]}"; do
  echo "===================================================================================================="
  echo " Building ${LANG}-en vocab with vocab_size=$VOCAB_SIZE and min_frequency=$MIN_FREQUENCY"
  echo "===================================================================================================="
  python build_vocab.py --lang "$LANG" --vocab_size "$VOCAB_SIZE" --min_frequency "$MIN_FREQUENCY"
done

echo "Done building vocabs!"