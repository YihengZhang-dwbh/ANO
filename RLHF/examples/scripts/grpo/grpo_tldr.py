# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import torch
from dataclasses import dataclass, field
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl import GRPOConfig, GRPOTrainer

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_model_path: str = field(
        default=None, 
        metadata={"help": "The path to the reward model."}
    )
    sft_model_path: str = field(
        default=None, 
        metadata={"help": "The path to the SFT model."}
    )
    dataset_num_proc: int = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        trust_remote_code=model_args.trust_remote_code,
    )
    training_args.model_init_kwargs = model_init_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_path = script_args.sft_model_path if script_args.sft_model_path else model_args.model_name_or_path
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_init_kwargs
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def filter_long_prompts(example):
        return len(tokenizer(example["prompt"])["input_ids"]) <= 512

    with PartialState().local_main_process_first():
        train_dataset = train_dataset.filter(filter_long_prompts, num_proc=script_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(filter_long_prompts, num_proc=script_args.dataset_num_proc)

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[script_args.reward_model_path],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
