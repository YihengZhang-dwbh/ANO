source activate ano_rl

id=0
time=$(date '+%Y-%m-%d-%H%M%S')
algo=PPO
/bin/echo "mujoco ${id} ${algo}"

CUDA_VISIBLE_DEVICES=$id python mujoco.py --algo ${algo} --cuda True > train_${algo}_$time.txt 2>&1
