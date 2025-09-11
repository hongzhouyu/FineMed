"""
If you see this, I wish you a good life! :)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from trl import SFTConfig, SFTTrainer
 


tokenizer = AutoTokenizer.from_pretrained('/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/yuhongzhou/rebuttal/FineMedLM', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/yuhongzhou/rebuttal/FineMedLM') # device_map='auto',自动分配模型到多个 GPU

print("-----Load SFT dataset-----")
dataset1 = load_from_disk("/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/yuhongzhou/rebuttal/dpo1")
print(dataset1)

print("-----train_test_split-----")
sampled_dataset = dataset1.shuffle(seed=42)
# sampled_dataset = dataset1
# 按比例划分为训练集和测试集 
# 获取训练集和测试集
train_dataset = sampled_dataset
# 打印信息以验证
# train_dataset, eval_dataset

args = SFTConfig(
    output_dir="/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/yuhongzhou/rebuttal/FineMedLM-sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,               # 每10步记录一次日志
    num_train_epochs=2,
    learning_rate=5e-6,                   # 基础学习率
    warmup_ratio=0.1,                    # 预热步骤数
    weight_decay=0.01,
    lr_scheduler_type="cosine",          # 使用余弦学习率调度器
    # gradient_checkpointing=True,
    seed=42,                         # 设置全局随机种子
    max_length=None,
    report_to="tensorboard",         # 日志会被记录到 output_dir 下的 runs/
    bf16=True,  # 启用 bf16
    # deepspeed="/code/SFT/ds_config.json",        # 指定 DeepSpeed 配置文件路径
    save_steps=80,
    save_total_limit=5,
    save_only_model=True,
    save_safetensors=True,
)

trainer = SFTTrainer(
    model=model, args=args, processing_class=tokenizer, train_dataset=train_dataset
)

print("-----Starting Training-----")
trainer.train()
# trainer.save_model(output_dir="/model/FineMedLM-lr1e-5/final")
# tokenizer.save_pretrained("/model/FineMedLM-lr1e-5/final")

print("-----SFT succeeded!-----")
