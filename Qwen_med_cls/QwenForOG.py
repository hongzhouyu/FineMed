from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

def get_completion(prompts, model, tokenizer=None, max_tokens=10, temperature=0, top_p=0.95):
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,  max_tokens=max_tokens)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, trust_remote_code=True, tensor_parallel_size=1, max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return outputs

# 定义处理单个输出的函数
def process_output(i, output):
    global Gynecology_count, Obstetrics_count, OtherDepartments_count, unknown_count
    generated_text = output.outputs[0].text
    # print(f"第 {i} 个输出：{generated_text}\n")
    if generated_text == 'Gynecology':
        with lock:  # 加锁以确保线程安全
            Gynecology_list.append(i)
            Gynecology_count += 1
    elif generated_text == 'Obstetrics':
        with lock:  # 加锁以确保线程安全
            Obstetrics_list.append(i)
            Obstetrics_count += 1
    elif generated_text == 'None':
        with lock:  # 加锁以确保线程安全
            OtherDepartments_list.append(i)
            OtherDepartments_count += 1
    else:
        with lock:  # 加锁以确保线程安全
            unknown_count += 1
            unknown_list.append(i)
            unknown_output.append(generated_text)

# 执行多线程处理
def process_outputs(outputs):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_output, i, output) for i, output in enumerate(outputs)]
        
        # 使用 tqdm 进度条，等待任务完成
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # 这里无需具体操作，只是等待每个任务完成

processed_dataset = load_from_disk("/home/ubuntu/test_disk/med_cls_prompt")

prompts = processed_dataset["prompt"][:100]
model = "/home/ubuntu/test_disk/qwen7b"
# model = "Qwen/Qwen2.5-32B-Instruct"

outputs = get_completion(prompts, model, temperature=0, top_p=0.9)

# 初始化共享变量
Gynecology_list = []
Obstetrics_list = []

OtherDepartments_list = []
unknown_list = []

unknown_output = []

Gynecology_count = 0
Obstetrics_count = 0

OtherDepartments_count = 0
unknown_count = 0

lock = threading.Lock()  # 用于线程同步的锁

# 假设 `outputs` 是已定义的数据
process_outputs(outputs)

ids = 0
# 保存为 .npy 文件
np.save(f'/home/ubuntu/disk/medical_data/ObstetricsandGynecology/Gynecology/Gynecology_list_{ids}.npy', Gynecology_list)
np.save(f'/home/ubuntu/disk/medical_data/ObstetricsandGynecology/Obstetrics/Obstetrics_list_{ids}.npy', Obstetrics_list)

np.save(f'/home/ubuntu/disk/medical_data/ObstetricsandGynecology/OtherDepartments/OtherDepartments_list_{ids}.npy', OtherDepartments_list)
np.save(f'/home/ubuntu/disk/medical_data/ObstetricsandGynecology/unknown/unknown_list_{ids}.npy', unknown_list)

# 保存为 JSON 文件
with open(f'/home/ubuntu/disk/medical_data/ObstetricsandGynecology/unknown/unknown_output_{ids}.json', 'w') as f:
    json.dump(unknown_output, f)
    
print(f"运行结束, 1 有 {Gynecology_count} 个目标")
print(f"运行结束, 2 有 {Obstetrics_count} 个目标")

print(f"运行结束，其他科室有 {OtherDepartments_count} 个目标")
print(f"运行结束，有 {unknown_count} 个目标没有被分类")

