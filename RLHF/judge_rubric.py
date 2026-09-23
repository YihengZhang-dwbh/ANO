# -*- coding: utf-8 -*-
"""Rubric-based LLM evaluation for RLHF checkpoints (pairwise, RM-free).

Why this exists: binary pairwise win/lose with N=100 has +-10pp CI and cannot
resolve SPO-vs-ANO-level differences. This script instead
  (1) generates completions with the EXACT protocol of judge.py (trl-lib/tldr
      test split, first N prompts, max_new_tokens=53, greedy when temp==0),
  (2) scores each completion INDEPENDENTLY on a 1-10 rubric (relevance,
      coherence, faithfulness, conciseness) via DeepSeek JSON mode -- no
      position bias, continuous resolution,
  (3) computes paired per-prompt deltas score(A)-score(B) with bootstrap CIs,
      Wilcoxon signed-rank, sign test, and a TOST equivalence test
      (90% CI inside (-delta, +delta) <=> "quality-equivalent" at margin delta).

Outputs (out_dir):
  completions_{PAIR}_t{temp}_n{N}_k{k}.jsonl   (cache; reuse on rerun)
  rubric_{PAIR}_t{temp}.json                   (all per-prompt scores + summary)
  rubric_{PAIR}_t{temp}_report.txt             (human-readable report)

Env: OPENAI_API_KEY must be set (DeepSeek key, same as judge.py uses).
Server usage:
  python judge_rubric.py --model_a_path <A> --model_b_path <B> \
      --num_examples 300 --temperature 0.0 --batch_size 16
"""
import argparse
import datetime
import gc
import json
import math
import os
import random
import re
import sys
import time
import concurrent.futures

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DIMS = ["relevance", "coherence", "faithfulness", "conciseness"]

RUBRIC_PROMPT = """You are evaluating a one-sentence summary (TL;DR) of a Reddit post. Rate the summary on four dimensions, each from 1 (worst) to 10 (best):

- relevance: captures the key content of the post.
- coherence: fluent, grammatical, readable.
- faithfulness: consistent with the source post, no invented facts.
- conciseness: information-dense, no redundancy.

Reply with a JSON object only, e.g. {{"relevance": 8, "coherence": 9, "faithfulness": 7, "conciseness": 8}}.

## Reddit post
{prompt}

## Summary
{completion}
"""


# ----------------------------- naming -----------------------------
def get_name(path):
    """'.../ano_tldr_0.2_10_-1.5[/checkpoint-12000]' -> 'ANO_0.2_10_-1.5'."""
    p = path.rstrip("/")
    base = os.path.basename(p)
    if "checkpoint" in base:
        base = os.path.basename(os.path.dirname(p))
    low = base.lower()
    if "sft" in low:
        return "SFT"
    algo = ("ANO" if "ano" in low else "PPO" if "ppo" in low else
            "SPO" if "spo" in low else "GRPO" if "grpo" in low else "MODEL")
    m = re.search(r"tldr_(.+)$", base)
    return f"{algo}_{m.group(1)}" if m else algo


# ----------------------------- generation (identical to judge.py) -----------------------------
def generate_responses(model_path, prompts, args, model_name, do_sample, temperature):
    print(f"\n[{model_name}] Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    outs_all = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"gen[{model_name}]"):
        bp = prompts[i:i + args.batch_size]
        inputs = tokenizer(bp, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=53,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)
        outs_all.append([t.strip() for t in texts])
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return outs_all  # list per batch of lists per prompt


# ----------------------------- rubric judge -----------------------------
class RubricJudge:
    def __init__(self, model="deepseek-chat", base_url="https://api.deepseek.com",
                 max_workers=16, retries=5):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                             base_url=base_url)
        self.model = model
        self.max_workers = max_workers
        self.retries = retries

    def _score_one(self, prompt, completion):
        content = RUBRIC_PROMPT.format(prompt=prompt, completion=completion)
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0, max_tokens=150,
                    response_format={"type": "json_object"})
                txt = r.choices[0].message.content
                m = re.search(r"\{.*\}", txt, re.S)
                d = json.loads(m.group(0))
                return {k: max(1, min(10, int(round(float(d.get(k, np.nan)))))) for k in DIMS}
            except Exception as e:
                if attempt == self.retries - 1:
                    print(f"[judge] failed after {self.retries} tries: {e}")
                    return {k: np.nan for k in DIMS}
                time.sleep(2 ** attempt + random.random())

    def score(self, jobs):
        """jobs: list of (idx, key, prompt, completion). Returns {(idx,key): dims-dict}."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._score_one, p, c): (i, k)
                    for (i, k, p, c) in jobs}
            for fut in tqdm(concurrent.futures.as_completed(futs),
                            total=len(futs), desc="judge"):
                i, k = futs[fut]
                results[(i, k)] = fut.result()
        return results


# ----------------------------- statistics -----------------------------
def bootstrap_ci(x, n_iter=10000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_iter, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.std(x, ddof=1) / math.sqrt(len(x))), \
        (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))), \
        (float(np.percentile(means, 5)), float(np.percentile(means, 95)))


def wilcoxon_signed_rank(d):
    try:
        from scipy.stats import wilcoxon
        d = d[~np.isnan(d)]
        if len(d) < 10 or np.all(d == 0):
            return np.nan
        return float(wilcoxon(d).pvalue)
    except Exception:
        return np.nan


def sign_test(d):
    d = d[~np.isnan(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return np.nan, pos, neg
    # two-sided binomial p via normal approx with continuity correction
    z = (min(pos, neg) + 0.5 - n / 2) / math.sqrt(n / 4)
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return float(p), pos, neg


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a_path", required=True)
    ap.add_argument("--model_b_path", required=True)
    ap.add_argument("--num_examples", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_samples", type=int, default=1,
                    help="samples per prompt when temperature > 0 (expected-quality mode)")
    ap.add_argument("--tost_delta", type=float, default=0.5,
                    help="equivalence margin on the 1-10 scale")
    ap.add_argument("--judge_model", default="deepseek-chat")
    ap.add_argument("--judge_base_url", default="https://api.deepseek.com")
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--out_dir", default="rubric_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    MA, MB = get_name(args.model_a_path), get_name(args.model_b_path)
    PAIR = "_vs_".join(sorted([MA, MB]))
    do_sample = args.temperature > 0
    K = args.num_samples if do_sample else 1

    # ---- prompts ----
    dataset = load_dataset("trl-lib/tldr", split="test")
    dataset = dataset.select(range(args.num_examples))
    prompts = list(dataset["prompt"])

    # ---- completions (cached) ----
    cache = os.path.join(args.out_dir,
                         f"completions_{PAIR}_t{args.temperature}_n{len(prompts)}_k{K}.jsonl")
    if os.path.exists(cache):
        print(f"reusing cached completions: {cache}")
        rows = [json.loads(l) for l in open(cache, encoding="utf-8")]
    else:
        print(f"generating completions -> {cache}")
        all_rows = [{"idx": i, "prompt": p, "samples": {}} for i, p in enumerate(prompts)]
        for mk, mp in [(MA, args.model_a_path), (MB, args.model_b_path)]:
            for k_pass in range(K):
                outs = generate_responses(mp, prompts, args, f"{mk}[{k_pass}]",
                                          do_sample, args.temperature)
                flat = [c for batch in outs for c in batch]
                for r, c in zip(all_rows, flat):
                    r["samples"].setdefault(mk, []).append(c)
        with open(cache, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows = all_rows

    # ---- judging ----
    jobs = []
    for r in rows:
        for mk in (MA, MB):
            for s_i, comp in enumerate(r["samples"][mk]):
                jobs.append((r["idx"], f"{mk}#{s_i}", r["prompt"], comp))
    random.seed(1234)
    random.shuffle(jobs)  # interleave models/prompts to cancel judge drift
    scores = RubricJudge(args.judge_model, args.judge_base_url,
                         args.max_workers).score(jobs)

    # ---- aggregate per prompt ----
    recs = []
    for r in rows:
        rec = {"idx": r["idx"], "prompt": r["prompt"]}
        for mk in (MA, MB):
            per_dim = {k: [] for k in DIMS}
            for s_i in range(len(r["samples"][mk])):
                d = scores.get((r["idx"], f"{mk}#{s_i}"))
                if d:
                    for k in DIMS:
                        per_dim[k].append(d[k])
            for k in DIMS:
                rec[f"{mk}_{k}"] = float(np.nanmean(per_dim[k]))
            rec[f"{mk}_overall"] = float(np.nanmean([rec[f"{mk}_{k}"] for k in DIMS]))
            rec[f"{mk}_len"] = int(np.mean([len(c) for c in r["samples"][mk]]))
        recs.append(rec)

    # ---- paired stats ----
    def delta(key):
        return np.array([a[f"{MA}_{key}"] - a[f"{MB}_{key}"] for a in recs], dtype=float)

    rep = []
    rep.append(f"Rubric evaluation  {MA} vs {MB}")
    rep.append(f"time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rep.append(f"prompts N={len(recs)}, gen temp={args.temperature}, "
               f"judge={args.judge_model} (independent 1-10 rubric, JSON mode)")
    rep.append("=" * 68)
    rep.append(f"{'dimension':<24}{MA:>10}{MB:>10}{'meanD':>9}{'95% CI':>22}")
    summary = {}
    for k in DIMS + ["overall"]:
        d = delta(k)
        mean, se, ci95, ci90 = bootstrap_ci(d)
        summary[k] = {"mean_delta": mean, "se": se, "ci95": ci95, "ci90": ci90,
                      "score_A": float(np.nanmean([a[f"{MA}_{k}"] for a in recs])),
                      "score_B": float(np.nanmean([a[f"{MB}_{k}"] for a in recs]))}
        rep.append(f"{k:<24}{summary[k]['score_A']:>10.3f}{summary[k]['score_B']:>10.3f}"
                   f"{mean:>9.3f}   [{ci95[0]:.3f}, {ci95[1]:.3f}]")
    d = delta("overall")
    mean, se, ci95, ci90 = bootstrap_ci(d)
    p_w = wilcoxon_signed_rank(d)
    p_s, n_pos, n_neg = sign_test(d)
    equiv = (ci90[0] > -args.tost_delta) and (ci90[1] < args.tost_delta)
    rep.append("=" * 68)
    rep.append(f"paired per-prompt delta (A-B), N_valid={int((~np.isnan(d)).sum())}")
    rep.append(f"Wilcoxon signed-rank p = {p_w:.4f} | sign test p = {p_s:.4f} "
               f"(A better: {n_pos}, B better: {n_neg})")
    rep.append(f"TOST equivalence: delta={args.tost_delta}, 90% CI=[{ci90[0]:.3f}, {ci90[1]:.3f}]"
               f" -> {'EQUIVALENT' if equiv else 'not established'}")
    for mk in (MA, MB):
        L = np.array([a[f"{mk}_len"] for a in recs], dtype=float)
        rep.append(f"mean completion chars [{mk}]: {L.mean():.0f} +- {L.std(ddof=1):.0f}")
    report = "\n".join(rep)
    print("\n" + report)

    out = {"model_a": MA, "model_b": MB, "n": len(recs), "temp": args.temperature,
           "judge": args.judge_model, "dims": DIMS,
           "summary": summary, "wilcoxon_p": p_w, "sign_p": p_s,
           "sign_counts": [n_pos, n_neg], "tost_delta": args.tost_delta,
           "tost_ci90": ci90, "tost_equivalent": equiv, "per_prompt": recs}
    jpath = os.path.join(args.out_dir, f"rubric_{PAIR}_t{args.temperature}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    rpath = os.path.join(args.out_dir, f"rubric_{PAIR}_t{args.temperature}_report.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved {jpath}\nsaved {rpath}")


if __name__ == "__main__":
    main()
