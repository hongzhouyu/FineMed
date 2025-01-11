from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from trl import SFTConfig, SFTTrainer
 

# 设置使用 4, 5, 6, 7 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

tokenizer = AutoTokenizer.from_pretrained('/home/ma-user/work/code_dev/cth/model/FineMedLM-s2-lr5e-6/checkpoint-506', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/home/ma-user/work/code_dev/cth/model/FineMedLM-s2-lr5e-6/checkpoint-506') # device_map='auto',自动分配模型到多个 GPU

messages_template = [
    {"role": "system", "content": "You are a helpful professional doctor."},
    {"role": "user", "content": "{prompt}"}
]

def create_messages(prompt):
    return [
        msg.copy() if msg["role"] == "system" 
        else {"role": "user", "content": prompt}
        for msg in messages_template
    ]
def create_prompt(example):
    message = create_messages(example["instruction"])
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        # 训练时不用加入 <|im_start|>assistant
        add_generation_prompt=False
    )
    example["prompt"] = text
    return example

def process_func(example):
    MAX_LENGTH = 1024    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(example["prompt"], add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['response']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("-----Load SFT dataset-----")
dataset1 = load_from_disk("/home/ma-user/work/code_dev/cth/Datasets/3StagesSFTData/InternalMedicine/Endocrinology")
print(dataset1)

print("-----train_test_split-----")
# 确定采样数量
sample_size = 11000
# s1 随机采样 20 万条数据, s2 -> 36000, s3 -> 11000
sampled_dataset = dataset1.shuffle(seed=42).select(range(sample_size))
# sampled_dataset = dataset1
# 按比例划分为训练集和测试集 
train_test_split = sampled_dataset.train_test_split(test_size=0.1, seed=42)
# 获取训练集和测试集
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
# 打印信息以验证
# train_dataset, eval_dataset

print("-----create prompt-----")
train_dataset = train_dataset.map(create_prompt)
eval_dataset = eval_dataset.map(create_prompt)

print("-----transfer to tokenized id-----")
# 检查是否已经有 pad_token
if tokenizer.pad_token is None:
    # 添加新的 pad_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # 调整模型的嵌入层大小
tokenized_train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)

args = SFTConfig(
    output_dir="/home/ma-user/work/code_dev/cth/model/FineMedLM-s3-lr5e-6",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_dir="/home/ma-user/work/code_dev/cth/model/logs", # TensorBoard日志文件保存路径
    logging_steps=10,               # 每10步记录一次日志
    eval_steps=30,
    num_train_epochs=1,
    eval_strategy="steps",  # 或 "epoch"
    save_steps=200,
    learning_rate=5e-6,                   # 基础学习率
    warmup_ratio=0.1,                    # 预热步骤数
    weight_decay=0.01,
    lr_scheduler_type="cosine",          # 使用余弦学习率调度器
    gradient_checkpointing=True,
    seed=42,                         # 设置全局随机种子
    bf16=True,  # 启用 bf16
    # deepspeed="/home/ma-user/work/code_dev/cth/code/SFT/ds_config.json",        # 指定 DeepSpeed 配置文件路径
    # 很重要！
    # gradient_checkpointing_kwargs={"use_reentrant":False}
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("-----Starting Training-----")
trainer.train()
# trainer.save_model(output_dir="/home/ma-user/work/code_dev/cth/model/FineMedLM-lr1e-5/final")
# tokenizer.save_pretrained("/home/ma-user/work/code_dev/cth/model/FineMedLM-lr1e-5/final")

print("-----SFT succeeded!-----")
