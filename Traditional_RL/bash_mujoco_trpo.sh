source activate ano_rl

id=0
time=$(date '+%Y-%m-%d-%H%M%S')
algo=TRPO

echo "Starting Job: Algo=${algo} GPU=${id} Time=${time}"

CUDA_VISIBLE_DEVICES=$id python mujoco.py \
    --algo ${algo} \
    --trpo-max-kl 0.01 \
    --trpo-cg-iters 10 \
    --trpo-damping 0.1 \
    --cuda True \
    > train_${algo}_${time}.txt 2>&1
