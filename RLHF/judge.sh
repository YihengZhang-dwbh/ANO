temp=0
step=500
num=100

source activate ano_trl

export CUDA_VISIBLE_DEVICES=0,1

ano=ano/checkpoint-$step
ano_f=ano/

ppo=ppo/checkpoint-$step
ppo_f=ppo/

python judge_tie.py \
    --model_a_path $ano \
    --model_b_path $ppo \
    --num_examples $num \
    --temperature $temp \
    --batch_size 16

