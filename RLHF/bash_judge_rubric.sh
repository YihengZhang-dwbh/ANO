# -*- bash -*-  Rubric evaluation driver -- paste-ready for the server.
# Pairs: ANO(0.2,10,-1.5) vs PPO(0.2), ANO vs SPO(0.2); decoding temps 0/0.7/1.0,
# 300 TL;DR test prompts, independent 1-10 rubric judging via DeepSeek.

#conda config --add envs_dirs /mnt/share/yiheng/miniconda3/envs
#source activate ano_trl

export CUDA_VISIBLE_DEVICES=3
export OPENAI_API_KEY="sk-your-deepseek-key"   # same key judge.py uses

BASE=/mnt/share/yiheng/TRL/models/minimal
checkpoint=12000

ano="$BASE/ano_tldr_0.2_10_-1.5/checkpoint-$checkpoint"
ano_f="$BASE/ano_tldr_0.2_10_-1.5"
spo_f="$BASE/spo_tldr_0.2"
ppo_f="$BASE/ppo_tldr_0.2"

num=300

# ---------- main: temps 0 / 0.7 / 1.0, ANO vs PPO and ANO vs SPO ----------
for temp in 0.0 0.7 1.0; do
    for pair in "$ano_f $ppo_f" "$ano_f $spo_f"; do
        set -- $pair
        python judge_rubric.py \
            --model_a_path "$1" \
            --model_b_path "$2" \
            --num_examples $num \
            --temperature $temp \
            --batch_size 16 \
            --tost_delta 0.5
    done
done

# ---------- optional: expected-quality mode (k samples per prompt, averages out decode noise) ----------
# for temp in 0.7 1.0; do
#     for pair in "$ano_f $ppo_f" "$ano_f $spo_f"; do
#         set -- $pair
#         python judge_rubric.py \
#             --model_a_path "$1" \
#             --model_b_path "$2" \
#             --num_examples $num \
#             --temperature $temp \
#             --num_samples 4 \
#             --batch_size 16 \
#             --tost_delta 0.5
#     done
# done

# ---------- optional: cross-judge robustness (e.g., another OpenAI-compatible endpoint) ----------
# python judge_rubric.py --model_a_path "$ano_f" --model_b_path "$ppo_f" \
#     --num_examples $num --temperature 0.0 --batch_size 16 \
#     --judge_model qwen-plus --judge_base_url https://dashscope.aliyuncs.com/compatible-mode/v1
