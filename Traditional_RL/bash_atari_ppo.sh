source activate ano_rl

id=0
time=$(date '+%Y-%m-%d-%H%M%S')
algo=PPO
/bin/echo "${id} ${algo}"

CUDA_VISIBLE_DEVICES=$id python atari.py --algo ${algo} --cuda True > train_${algo}_$time.txt 2>&1
