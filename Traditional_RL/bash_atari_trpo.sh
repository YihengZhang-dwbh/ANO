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
    --trpo-max-kl 0.01 \
    --trpo-cg-iters 10 \
    --trpo-damping 0.1 \
    --update-epochs 4 \
    --cuda True \
    > train_${algo}_${time}.txt 2>&1
