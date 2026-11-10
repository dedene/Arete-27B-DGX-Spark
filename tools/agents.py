#!/usr/bin/env python3
"""Many agents at once: does a follow-up find its conversation still cached?

Each agent sends its own long prompt (a distinct document, ~17K tokens by default), and once every agent has its first
reply, all of them send a follow-up on the same conversation together. A follow-up whose prompt state survived starts
almost at once; one that was evicted prefills the whole conversation again. This is the shape of TensorFold 1.2.6's
headline test (#2699). Greedy and thinking off by default, so both servers do the same work. Stdlib only.

    python3 tools/agents.py --agents 4,8,16            # against http://127.0.0.1:8888
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

THINK = False

WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but have an they "
         "you were her she there been one all we their has would when if so no will more can out up into do any your "
         "what some about these other than then them only could may its also two after first new most over such well "
         "where through back years much before must between under never same last while might great since against "
         "river garden window market letter station winter harbour engine copper ledger meadow lantern orchard bridge "
         "council invoice server kitchen tunnel signal pocket shelter thunder canvas velvet anchor beacon compass").split()


def document(agent: int, words: int, salt: int) -> str:
    """Distinct per agent and per round, so no earlier round's cached state helps."""
    rnd = random.Random(f"{salt}-{agent}")
    parts = [f"Field notes of expedition {salt}-{agent}, kept by surveyor {rnd.randrange(10**6):06d}."]
    for section in range(1, 10**6):
        if len(" ".join(parts).split()) >= words:
            break
        body = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(60, 120)))
        parts.append(f"\n\nSection {section} (code {rnd.randrange(10**5):05d}): {body}.")
    return "".join(parts)


def stream(url: str, model: str, messages: list, max_tokens: int, timeout: float) -> dict:
    req = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": THINK}}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    t0, first, text, usage = time.monotonic(), None, [], {}
    with urllib.request.urlopen(r, timeout=timeout) as f:
        for raw in f:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    first = first or time.monotonic()
                    text.append(piece)
    end = time.monotonic()
    first = first or end
    out = usage.get("completion_tokens") or 0
    return {"ttft": first - t0, "total": end - t0, "text": "".join(text), "prompt": usage.get("prompt_tokens"),
            "decode": (out - 1) / (end - first) if out > 1 and end > first else None}


def run(args, n: int) -> dict:
    words = int(args.prompt_tokens / 1.35)
    convs = [[{"role": "system", "content": "You are a careful research assistant. Answer briefly."},
              {"role": "user", "content": document(i, words, n * 1000 + args.round) + "\n\nSummarise section 3 in two sentences."}]
             for i in range(n)]
    results = {"first": [None] * n, "follow": [None] * n}

    def turn(kind: str, i: int):
        try:
            results[kind][i] = stream(args.url, args.model, convs[i], args.max_tokens, args.timeout)
        except Exception as e:  # noqa: BLE001  (report it, keep the others going)
            results[kind][i] = {"error": repr(e)[:200]}

    for kind in ("first", "follow"):
        if kind == "follow":
            for i in range(n):
                prev = results["first"][i]
                convs[i] += [{"role": "assistant", "content": prev.get("text", "")},
                             {"role": "user", "content": "Now name the code of section 5 and one word it repeats."}]
        threads = [threading.Thread(target=turn, args=(kind, i)) for i in range(n)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results[kind + "_wall"] = time.monotonic() - t0
    return results


def summary(rs: list) -> str:
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return "all failed"
    tt = sorted(r["ttft"] for r in ok)
    dec = [r["decode"] for r in ok if r["decode"]]
    p90 = tt[min(len(tt) - 1, int(0.9 * len(tt)))]
    return (f"ttft median {statistics.median(tt):6.1f} s  p90 {p90:6.1f} s  max {tt[-1]:6.1f} s  "
            f"decode median {statistics.median(dec) if dec else 0:5.1f} tok/s  prompt ~{ok[0]['prompt']} tok"
            + (f"  ERRORS {len(rs) - len(ok)}" if len(ok) < len(rs) else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="Arete-27B")
    p.add_argument("--agents", default="4,8,16", help="comma-separated agent counts, run in this order")
    p.add_argument("--prompt-tokens", type=int, default=17000)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--round", type=int, default=0, help="change it to rerun on a warm server with fresh prompts")
    p.add_argument("--json", help="also write every request's numbers to this file")
    p.add_argument("--think", action="store_true", help="thinking on, as Hermes runs (earlier reasoning is dropped from history)")
    args = p.parse_args()
    global THINK
    THINK = args.think
    report = {}
    for n in (int(x) for x in args.agents.split(",")):
        r = run(args, n)
        report[n] = r
        print(f"{n:3} agents  first turn  {summary(r['first'])}  (all done in {r['first_wall']:.0f} s)", flush=True)
        print(f"{n:3} agents  follow-up   {summary(r['follow'])}  (all done in {r['follow_wall']:.0f} s)", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({n: {k: ([{kk: vv for kk, vv in x.items() if kk != "text"} for x in v] if isinstance(v, list)
                               else v) for k, v in r.items()} for n, r in report.items()}, f, indent=1)


if __name__ == "__main__":
    main()
