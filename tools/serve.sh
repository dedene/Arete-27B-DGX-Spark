#!/usr/bin/env bash
# Start TensorFold's native server on the pinned Arete-27B MLX 4-bit checkpoint. Every engine flag lives here.
set -euo pipefail

model=/models/arete-27b-mlx-4bit
drafter=/models/dflash2
[[ -f $model/config.json ]] || { echo "no checkpoint at $model" >&2; exit 3; }

[[ $ENABLE_THINKING =~ ^[01]$ ]] || { echo "ENABLE_THINKING is 0 or 1, not $ENABLE_THINKING" >&2; exit 2; }
[[ $KV_CACHE =~ ^(bf16|fp8)$ ]] || { echo "KV_CACHE is bf16 or fp8, not $KV_CACHE" >&2; exit 2; }
[[ $VISION =~ ^[01]$ ]] || { echo "VISION is 0 or 1, not $VISION" >&2; exit 2; }

args=(serve "$model" --name "$SERVED_NAME" --host "$HOST" --port "$PORT" --context "$CONTEXT" --parallel "$PARALLEL"
      --max-tokens "$MAX_TOKENS" --temperature "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --kv-cache "$KV_CACHE")
if [[ $ENABLE_THINKING == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
case $SPEC in
  dflash2) [[ -f $drafter/model.safetensors ]] || { echo "no drafter at $drafter" >&2; exit 3; }
           args+=(--drafter "$drafter") ;;
  none) args+=(--no-drafts) ;;
  *) echo "SPEC is dflash2 or none, not $SPEC" >&2; exit 2 ;;
esac
[[ $VISION == 1 ]] && args+=(--vision)
[[ -n ${TENSORFOLD_API_KEY:-} ]] || unset TENSORFOLD_API_KEY

echo "tensorfold-native ${args[*]} $*"
exec tensorfold-native "${args[@]}" "$@"
