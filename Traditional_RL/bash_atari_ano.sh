source activate ano_rl

id=0
time=$(date '+%Y-%m-%d-%H%M%S')
algo=ANO
clip_ratio=(0.1 0.1)
strr=$(printf '%s,' "${clip_ratio[@]}")
strr=${strr%,}
/bin/echo "${id} ${algo} ${clip_ratio[@]}"

CUDA_VISIBLE_DEVICES=$id python atari.py --algo ${algo} --epsilons "${clip_ratio[@]}" --cuda True > train_${algo}_${strr}_$time.txt 2>&1

