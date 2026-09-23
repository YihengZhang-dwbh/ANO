# ANO — RLHF Implementation (TRL fork)

On-policy RLHF trainers and the LLM-judge evaluation pipeline used in the
paper's Reddit TL;DR experiments (Pythia-1B policy + reward model, 1M episodes).

## What's here

- **On-policy trainers**: `experimental/ano`, `experimental/papo`,
  `experimental/spo`, `experimental/ppo` (mirrored under `trl/experimental/*`
  for direct `import` use). ANO exposes the same three knobs as the paper:
  trust-region boundary `eps`, `kappa_plus` (default 10), `kappa_minus`
  (default −1.5).
- **LLM-judge evaluation**: `judge_rubric.py` / `bash_judge_rubric.sh`
  (four-dimension 1–10 rubric, 300 held-out prompts, greedy decoding; raw
  judge outputs are kept in `rubric_results/`).

## Quick start

```bash
git clone <YOUR_REPO_URL>
cd ANO/RLHF

conda env create -f ano_trl.yaml
conda activate ano_trl

bash bash_ano.sh            # ANO training
# baselines:
bash bash_ppo.sh
bash bash_grpo.sh

bash bash_judge_rubric.sh   # evaluate trained checkpoints with the LLM judge
```

The judge is an external LLM accessed through an OpenAI-compatible API;
configure the endpoint and key via environment variables (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`).

## Citation

```bibtex
@inproceedings{Anonymous2027ano,
  title     = {ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields},
  author    = {Anonymous authors},
  booktitle = {Under review},
  year      = {2027}
}
```

## Acknowledgements

This directory builds upon [TRL](https://github.com/huggingface/trl)
(Transformer Reinforcement Learning); please refer to its license and cite it
if you build upon its work.
