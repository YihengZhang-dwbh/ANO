source activate ano_rl 

id=0
time=$(date '+%Y-%m-%d-%H%M%S')
algo=TRPO

echo "Starting Job: Algo=${algo} GPU=${id} Time=${time}"


CUDA_VISIBLE_DEVICES=$id python atari.py \
    --algo ${algo} \
    --total-timesteps 6000000 \
    --num-envs 16 \
    --learning-rate 2.5e-4 \
    --max-kl 0.01 \
    --cg-iters 10 \
    --cg-damping 0.1 \
    --trpo-update-epochs 4 \
    --cuda True \
    > train_${algo}_${time}.txt 2>&1
