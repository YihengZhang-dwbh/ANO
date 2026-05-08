temp=0
num=100

source activate ano_trl

export CUDA_VISIBLE_DEVICES=0,1

ano="your_anon_checkpoint-step"

ppo="your_ppo_checkpoint-step"

python judge.py \
    --model_a_path $ano \
    --model_b_path $ppo \
    --num_examples $num \
    --temperature $temp \
    --batch_size 16
