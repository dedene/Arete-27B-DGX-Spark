#!/usr/bin/env bash
# Fetch the pinned Arete-27B MLX 4-bit weights, Nous's own tokenizer and the DFlash2 drafter into DEST (default /models).
# Resumable, every file sha256 checked. A later start is a no-op once each file's stamp matches.
set -euo pipefail

dest=${DEST:-/models}
weights=${MODEL_ID:-Duckets/Arete-27B-MLX-4bit}
weights_rev=${MODEL_REVISION:-bbbb47db81f9e057ab523b810cd7893384b1699c}
nous=NousResearch/Arete-27B
nous_rev=108f0afbdcfaa1ceeda4a635004304dfff6d4da2
drafter=z-lab/Qwen3.8-27B-DFlash2
drafter_rev=50307d4c4cde6860d4eee73e2547cd786fe8e8a4

curl_fetch() {
  local url=$1 path=$2
  local -a args=(-fL --retry 8 --retry-delay 2 --retry-all-errors -C - --output "$path")
  [[ -n ${HF_TOKEN:-} ]] && args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  curl "${args[@]}" "$url"
}

sum() { sha256sum "$1" | awk 'NR==1 { print $1 }'; }

# fetch <repo> <revision> <file in repo> <folder under dest> <file there> <sha256>
fetch() {
  local repo=$1 rev=$2 name=$3 dir=$dest/$4 out=$5 expect=$6
  local path=$dir/$out stamp=$dir/.$out.sha256 url="https://huggingface.co/${repo}/resolve/${rev}/${name}"
  mkdir -p "$dir"
  if [[ -f $stamp && $(<"$stamp") != "$rev $expect" ]]; then
    echo "revision changed, downloading $out again"
    rm -f "$path" "$stamp"
  fi
  if [[ -f $path && -f $stamp ]]; then
    echo "present $out"
    return
  fi
  # A complete file without a stamp (copied in, or an interrupted check) is checked rather than fetched again.
  if [[ ! -f $path ]] || [[ $(sum "$path") != "$expect" ]]; then
    echo "downloading $out"
    curl_fetch "$url" "$path"
    if [[ $(sum "$path") != "$expect" ]]; then
      echo "checksum mismatch for $out, downloading again" >&2
      rm -f "$path"
      curl_fetch "$url" "$path"
      [[ $(sum "$path") == "$expect" ]] || { echo "checksum mismatch for $out" >&2; exit 1; }
    fi
  fi
  printf '%s %s\n' "$rev" "$expect" >"$stamp"
  echo "verified $out"
}

m=arete-27b-mlx-4bit
w() { fetch "$weights" "$weights_rev" "$1" "$m" "$1" "$2"; }
w model-00001-of-00003.safetensors 8add69de1a3317fac78bff2a32b7ce8dd515b758be62d302fbdee66cc051d37d
w model-00002-of-00003.safetensors f817c92f1235c7d68c0766bff9a9ef4ca899f588370a6aa330dcfc9d2ff19965
w model-00003-of-00003.safetensors 30ab16422e4d0edd0ad8aff36b30cfa5614eba7688724e9bd11e735b563c4858
w model.safetensors.index.json 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8
w config.json 0fe325c8c7fce99d195afd1c4a0a5256c8720c3dd33c30bc7c7679c6841afb58
w generation_config.json 0ceb40dd22663ead8d1122e21c5bec9fc703da974485a7e493b301e382fa7f57
w chat_template.jinja c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041
w preprocessor_config.json 3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2
w processor_config.json 45fc17c8dd2474af6b493b52483c26c0584b0082d368c480f9fa611e73070040
w video_preprocessor_config.json 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13
w vocab.json ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003

# The MLX conversion rewrote the pre-tokenizer's regex. Nous's tokenizer is the one the model was trained with.
n() { fetch "$nous" "$nous_rev" "$1" "$m" "$1" "$2"; }
n tokenizer.json 06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523
n tokenizer_config.json 7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8
n merges.txt a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d

d() { fetch "$drafter" "$drafter_rev" "$1" dflash2 "$1" "$2"; }
d config.json 873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980
d model.safetensors 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
exit 0
