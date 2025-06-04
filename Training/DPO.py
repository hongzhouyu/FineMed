# train_dpo.py
from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

model_name_or_path = "FineMedLM-sft/checkpoint-200"  # 或者你的本地路径（如 "./qwen-32b-checkpoints"）

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
train_dataset = load_from_disk("/dpo2")
print(train_dataset)
print(train_dataset[0])

training_args = DPOConfig(output_dir="FineMedLM-o1", 
                          logging_steps=10,
                        #   eval_steps=30,
                          num_train_epochs=1,
                          per_device_train_batch_size=1,
                          gradient_accumulation_steps=16,
                          learning_rate=1e-7,                   # 基础学习率
                          warmup_ratio=0.1,                    # 预热步骤数
                          weight_decay=0.01,
                          lr_scheduler_type="cosine",          # 使用余弦学习率调度器
                          gradient_checkpointing=True,
                          seed=42,                         # 设置全局随机种子
                          bf16=True,  # 启用 bf16
                          max_prompt_length=None,
                          max_length=None,
                          # deepspeed="/code/SFT/ds_config.json",        # 指定 DeepSpeed 配置文件路径
                          )
trainer = DPOTrainer(model=model, ref_model=ref_model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()

