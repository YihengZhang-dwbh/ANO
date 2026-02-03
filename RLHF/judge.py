import torch
import gc
import numpy as np
import concurrent.futures
import logging
from dataclasses import dataclass, field
from tqdm import tqdm
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
import datetime
import re

try:
    from judges import DeepSeekPairwiseJudge, HfPairwiseJudge, OpenAIPairwiseJudge
except ImportError:
    from trl.experimental.judges import DeepSeekPairwiseJudge, HfPairwiseJudge, OpenAIPairwiseJudge

class DeepSeekPairwiseJudgeWithTie(DeepSeekPairwiseJudge):
    TIE_SUPPORT_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results.
- Reply "0" if Model 0 is better.
- Reply "1" if Model 1 is better.
- Reply "2" if both models are of equal quality (Tie).

Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers (0, 1, or 2) and nothing else.
'''

    def __init__(self, **kwargs):
        super().__init__(system_prompt=self.TIE_SUPPORT_PROMPT, **kwargs)

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            messages = [{"role": "user", "content": content}]
            try:
                completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1)
                response = completion.choices[0].message.content.strip()
                
                if response in ["0", "1", "2"]:
                    return int(response)
                else:
                    return -1
            except Exception as e:
                print(f"API Error: {e}")
                return -1

        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        if shuffle_order:
            final_ranks = []
            for r, flip in zip(ranks, flip_mask):
                if r == -1 or r == 2: 
                    final_ranks.append(r)
                elif flip:
                    final_ranks.append(1 - r)
                else:
                    final_ranks.append(r)
            ranks = final_ranks

        return ranks

@dataclass
class ScriptArguments:
    model_a_path: str = field(default="/home/yiheng/trl/models/minimal/ano_r", metadata={"help": "Path to Model A."})
    model_b_path: str = field(default="/home/yiheng/trl/models/minimal/ppo_r", metadata={"help": "Path to Model B."})
    judge_model: str = field(default="deepseek", metadata={"help": "Judge model type."})
    num_examples: int | None = field(default=100, metadata={"help": "Number of examples to evaluate."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    temperature: float = field(default=0.7, metadata={"help": "Temperature for generation."})

def generate_responses(model_path, prompts, args, config0, model_name="Model"):
    print(f"\n🔄 [{model_name}] Loading from: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"✅ [{model_name}] Loaded successfully.")

    completions = []
    print(f"🚀 [{model_name}] Generating {len(prompts)} responses...")
    
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=53,
                do_sample=config0["do_sample"],
                temperature=config0.get("temperature", 1.0),
                pad_token_id=tokenizer.eos_token_id
            )
        batch_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        completions.extend([t.strip() for t in batch_texts])

    print(f"🧹 [{model_name}] Cleaning up memory...")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return completions

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    def get_name(path):  
        suffix = "_F" if "checkpoint" not in path else ""
        if "sft" in path: return "SFT"
        elif "gano" in path: return "GANO"+suffix
        elif "ano_0.3_0.05_tldr_2026-01-20-053736" in path: return "ANO_0.3"+suffix
        elif "ano" in path: return "ANO"+suffix
        elif "ppo" in path: return "PPO"+suffix
        elif "grpo" in path: return "GRPO"+suffix
        elif "spo" in path: return "SPO"+suffix
        raise ValueError(f"Cannot determine model name from path: {path}")
    
    MA = get_name(script_args.model_a_path)
    MB = get_name(script_args.model_b_path)

    M_STR = "_vs_".join(sorted([MA, MB]))

    if script_args.temperature == 0:
        config0 = {"target": "test", "do_sample": False}
    else:
        config0 = {"target": "test", "do_sample": True, "temperature": script_args.temperature}
    
    upper_target = config0["target"].capitalize()
    log_filename = f'evaluation_{M_STR}_{upper_target}_tie_log.txt'
    file = open(log_filename, 'a', encoding='utf-8')

    strt = f"{config0}\n📚 Loading Dataset ({upper_target} Split, {script_args.num_examples} samples)..."
    print(strt); file.write("\n" + strt)
    
    dataset = load_dataset("trl-lib/tldr", split=config0["target"])
    if script_args.num_examples:
        dataset = dataset.select(range(script_args.num_examples))
    prompts = dataset["prompt"]

    completions_a = generate_responses(script_args.model_a_path, prompts, script_args, config0, MA)
    completions_b = generate_responses(script_args.model_b_path, prompts, script_args, config0, MB)

    strt = f"\n⚖️  Initializing Judge: {script_args.judge_model} (With Tie Support)"
    print(strt); file.write(strt)
    
    if "deepseek" in script_args.judge_model.lower():
        judge = DeepSeekPairwiseJudgeWithTie()
    elif "gpt" in script_args.judge_model.lower():
        judge = OpenAIPairwiseJudge(script_args.judge_model)
    else:
        judge = HfPairwiseJudge(script_args.judge_model)

    strt = "\n⚡️ Judging pairs..."
    print(strt); file.write(strt)
    
    completion_pairs = [[c_a, c_b] for c_a, c_b in zip(completions_a, completions_b)]
    
    with open(f'completion_{M_STR}_{upper_target}_tie_pairs.txt', 'a', encoding='utf-8') as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"\n\n{'='*20} NEW TEST RUN: {current_time} {'='*20}\n"
        header += f"Model A ({MA}): {script_args.model_a_path}\n"
        header += f"Model B ({MB}): {script_args.model_b_path}\n"
        header += f"Judge Model: {script_args.judge_model}\n"
        header += f"Temperature: {script_args.temperature}\n"
        header += f"Data Split: {config0['target']} (N={len(completion_pairs)})\n"
        header += f"{'='*65}\n\n"
        
        f.write(header)
        for i, pair in enumerate(completion_pairs):
            f.write(f"--- Pair {i} ---\n")
            f.write(f"Prompt: {prompts[i]}\n")
            f.write(f"[{MA}]: {pair[0]}\n")
            f.write(f"[{MB}]: {pair[1]}\n")
            f.write("\n")

    best_idxs = judge.judge(prompts, completion_pairs, shuffle_order=True)

    wins_ma = best_idxs.count(0)
    wins_mb = best_idxs.count(1)
    ties = best_idxs.count(2)
    errors = best_idxs.count(-1)
    total_valid = len(best_idxs) - errors

    pct_ma = wins_ma / total_valid if total_valid else 0
    pct_mb = wins_mb / total_valid if total_valid else 0
    pct_tie = ties / total_valid if total_valid else 0

    strt = "\n" + "="*60 + "\n" 
    strt += f"📊  Evaluation Results (N={len(best_idxs)})\n"
    strt += f"Model A ({MA}): {script_args.model_a_path}\n"
    strt += f"Model B ({MB}): {script_args.model_b_path}\n"
    strt += f"Generation Config: Temp={script_args.temperature}\n"
    strt += "-" * 60 + "\n"
    strt += f"Model A ({MA}): {wins_ma:<5} ({pct_ma:.1%})  [WIN]\n"
    strt += f"Model B ({MB}): {wins_mb:<5} ({pct_mb:.1%})  [LOSS]\n"
    strt += f"Ties           : {ties:<5}     ({pct_tie:.1%})  [TIE]\n"
    strt += f"Errors         : {errors}\n"
    strt += "-" * 60 + "\n"
    
    strt += ">>> Data for Stacked Bar Chart <<<\n"
    strt += f"Category: [Win ({MA}), Tie, Loss ({MB})]\n"
    strt += f"Values  : [{pct_ma:.3f}, {pct_tie:.3f}, {pct_mb:.3f}]\n"

    tie_broken_wr = (wins_mb + 0.5 * ties) / total_valid if total_valid else 0
    strt += f"\n🏆 {MB} Tie-broken Win Rate vs {MA}: {tie_broken_wr:.2%}\n"

    strt += "="*60 + "\n"
    print(strt)
    file.write(strt)
    file.close()
