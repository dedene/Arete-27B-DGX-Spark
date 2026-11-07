# Arete-27B on a single DGX Spark (TensorFold, Docker Compose)

Serve [Arete-27B](https://huggingface.co/NousResearch/Arete-27B), Nous Research's fine-tune of Qwen3.8-27B, on one NVIDIA DGX Spark. One command builds the image, downloads the pinned MLX 4-bit weights and the DFlash2 drafter, and starts an OpenAI-compatible API at about 45 tok/s on mixed work and 123 tok/s across 4 requests. Model shape: 27B dense, 64 layers, 262,144-token window, text and images. About 20 GB on disk.

**Hardware:** DGX Spark (GB10, sm_121, aarch64), driver 580, CUDA 13, Docker with the NVIDIA Container Toolkit and Compose v2. About 45 GB free disk for the weights, drafter and image. The running server shows 85.6 GB of GPU memory at the 262,144-token window with 4 streams.

## Quickstart

```bash
git clone https://github.com/dedene/Arete-27B-DGX-Spark.git && cd Arete-27B-DGX-Spark
docker compose up -d --wait     # build, download 20 GB (first run only), load, wait until healthy
tools/check.py                  # thinking switch, tool call, streaming, Dutch, an image
docker compose down
```

The API is at `http://<spark-ip>:8888/v1`, model `Arete-27B`.

```bash
curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Arete-27B",
  "messages": [{"role": "user", "content": "Vat deze mail samen in twee zinnen."}]
}'
```

Thinking is on by default, the template's default. The trace comes back separately from the answer. To turn it off for one request, send `"chat_template_kwargs": {"enable_thinking": false}`.

## How it is put together

- `Dockerfile`: the TensorFold 1.2.3 release archive, sha256 checked, on MiaAI-Lab's published image. That image already has the NVIDIA PyTorch base (cuBLASLt for vision) and the JPEG and WebP decoders.
- `compose.yaml`: `download` (one-shot) then `arete` (the server, with a healthcheck).
- `tools/download.sh`: fetches each pinned file and checks its sha256. Resumable.
- `tools/serve.sh`: every engine flag.
- `tools/check.py`: functional checks, standard library only.

The weights come from [Duckets/Arete-27B-MLX-4bit](https://huggingface.co/Duckets/Arete-27B-MLX-4bit), a third-party conversion. Its tensor layout matches `TensorFold/Qwen3.8-27B-MLX-4bit`, and its norm tensors match Nous's weights once you allow for how the conversion stores them. Its `tokenizer.json` does not match: the conversion rewrote the pre-tokenizer's regex, so `download.sh` takes the tokenizer files from Nous.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT` | `262144` | Window per request |
| `PARALLEL` | `4` | Requests decoded together; a cap, because streams share one KV pool |
| `MAX_TOKENS` | `32768` | Longest reply |
| `ENABLE_THINKING` | `1` | Server default; a request can override it |
| `SPEC` | `dflash2` | `dflash2` or `none`. Drafts are verified, so this changes speed, not replies |
| `KV_CACHE` | `bf16` | `bf16` is exact; `fp8` is smaller and lossy |
| `VISION` | `1` | Image input |
| `TEMP` `TOP_P` `TOP_K` | `1.0` `0.95` `20` | Server sampling defaults, from `generation_config.json` |
| `SERVED_NAME` `HOST` `PORT` | `Arete-27B` `0.0.0.0` `8888` | Name and address |
| `API_KEY` | empty | Set to require a bearer token |
| `MODEL_ID` `MODEL_REVISION` | pinned | Source of the weights |
| `TENSORFOLD_IMAGE` `MODEL_DIR` `KERNEL_CACHE` `HF_TOKEN` `RESTART` | see `.env.example` | Image, data folders, restart policy |

## Notes

- Port 8888 clashes with other recipes on the same Spark. Stop those first.
- The API is open on the LAN unless `API_KEY` is set.
- DFlash2 was trained on base Qwen3.8-27B, not on Arete. The kit measured speed, but not how often drafts land on this fine-tune.

## As an agent: benchmark results

Measured on 2026-11-07 with spark-bench-kit, tier `office`, thinking on, temperature 0, 4 requests at once. The comparison is Qwen3.8 Flash Next as it runs daily here (TensorFold 1.1.4, NVIDIA FP4, thinking on) and Saluki 27B (a 2-bit Qwen3.8-27B on llama.cpp, thinking on). Numbers are counts or percentages with 95% intervals. Attack rows: lower is better.

| | Arete-27B | Qwen3.8 Flash Next | Saluki 27B |
|---|---:|---:|---:|
| E-mail attacks: action executed | 0/113 (0-3%) | 0/113 (0-3%) | 8/113 (4-13%) |
| E-mail attacks: user data disclosed, lower bound | 0/113 (0-3%) | 0/113 (0-3%) | 2/113 (0-6%) |
| All attacks: action executed, as scored | 3/200 (1-4%) | 1/200 (0-3%) | 29/200 (10-20%) |
| All attacks: user data disclosed, lower bound | 0/200 (0-2%) | 0/200 (0-2%) | 13/200 (4-11%) |
| AgentDojo tasks done, no attack | 82.3 (73-89, n=96) | 88.5 (81-93, n=96) | 75.0 (65-83, n=96) |
| AgentDojo tasks done under attack | 77.0 (71-82, n=200) | 85.5 (80-90, n=200) | 64.5 (58-71, n=200) |
| BFCL single-turn | 87.2 (84-90, n=500) | 85.8 (82-89, n=500) | 86.8 (84-89, n=500) |
| Scripted agent loop, steps per minute | 6.7 | 9.1 | 10.1 |

We read the 3 executed attacks in the Arete column and none is the agent acting on an attack. Two are travel samples where the reply warned about the injected hotel and refused it, and AgentDojo counted the hotel's name. The third posted the user's own text to a Slack channel whose name holds the injection.

Task scores are harder to read. Of the 19 attacked tasks Qwen finished and Arete did not, 10 are correct answers that fail AgentDojo's check because Arete called `get_unread_emails`, which marks mail as read. Counting those 10 as done gives 164/200 (82%), and the paired gap to Qwen is no longer clear (9 against 2, p = 0.065).

The loops are a real problem. Arete repeated the same write action until the message limit: up to 329 `create_calendar_event` calls in one task, with and without an attack. That happened in 2 of 200 attacked tasks and 2 of 96 clean tasks. One attacked task sent the same Slack message 42 times, and two clean tasks and one attacked task looped on `search_files`. Do not give this model write tools without a limit on repeated calls. The runs used temperature 0, and the loops have not been tested at the server default of 1.0.

Speed, measured separately from the scores: 44.5 tok/s weighted, 33.6 on Dutch prose, 112.5 on a tool call, 1,200-1,300 tok/s prefill, 122.7 tok/s across 4 requests. The office run took 3.1 hours.

This is one English attack template on one mailbox, so a clean count does not make the model safe. Multi-turn tool use, instruction following and Dutch tasks did not run in this tier. Base Qwen3.8-27B has not been run on this setup, so these numbers cannot say what the fine-tune changed.

## Credits

Arete-27B by [Nous Research](https://nousresearch.com/arete), Apache-2.0, on Qwen3.8-27B by Alibaba. MLX conversion by Duckets. Engine: [TensorFold](https://github.com/ashhart/TensorFold). Drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). Base image by [MiaAI-Lab](https://github.com/MiaAI-Lab).

## License

MIT for this repo. The weights are under their own licence.
