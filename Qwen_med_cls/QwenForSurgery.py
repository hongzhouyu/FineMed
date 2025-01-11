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
    llm = LLM(model=model, tokenizer=tokenizer, trust_remote_code=True, tensor_parallel_size=4, max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return outputs

# 定义处理单个输出的函数
def process_output(i, output):
    global Gastrointestinal_count, Hepatobiliary_count, Urology_count, Cardiovascular_count, Thoracic_count, Orthopedic_count, Neurosurgery_count, BurnsandPlastic_count, Thyroid_count, Breast_count, OtherDepartments_count, unknown_count
    generated_text = output.outputs[0].text
    # print(f"第 {i} 个输出：{generated_text}\n")
    if generated_text == 'Gastrointestinal Surgery':
        with lock:  # 加锁以确保线程安全
            Gastrointestinal_list.append(i)
            Gastrointestinal_count += 1
    elif generated_text == 'Hepatobiliary Surgery':
        with lock:  # 加锁以确保线程安全
            Hepatobiliary_list.append(i)
            Hepatobiliary_count += 1
    elif generated_text == 'Urology':
        with lock:  # 加锁以确保线程安全
            Urology_list.append(i)
            Urology_count += 1
    elif generated_text == 'Cardiovascular Surgery':
        with lock:  # 加锁以确保线程安全
            Cardiovascular_list.append(i)
            Cardiovascular_count += 1
    elif generated_text == 'Thoracic Surgery':
        with lock:  # 加锁以确保线程安全
            Thoracic_list.append(i)
            Thoracic_count += 1
    elif generated_text == 'Orthopedic Surgery':
        with lock:  # 加锁以确保线程安全
            Orthopedic_list.append(i)
            Orthopedic_count += 1
    elif generated_text == 'Neurosurgery':
        with lock:  # 加锁以确保线程安全
            Neurosurgery_list.append(i)
            Neurosurgery_count += 1
    elif generated_text == 'Burns and Plastic Surgery':
        with lock:  # 加锁以确保线程安全
            BurnsandPlastic_list.append(i)
            BurnsandPlastic_count += 1
    elif generated_text == 'Thyroid Surgery':
        with lock:  # 加锁以确保线程安全
            Thyroid_list.append(i)
            Thyroid_count += 1
    elif generated_text == 'Breast Surgery':
        with lock:  # 加锁以确保线程安全
            Breast_list.append(i)
            Breast_count += 1
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

processed_dataset = load_from_disk("/home/ma-user/work/code_dev/cth/data/med_cls/med_cls_prompt")

prompts = processed_dataset["prompt"][:100]
model = "/home/ubuntu/test_disk/qwen7b"
# model = "Qwen/Qwen2.5-32B-Instruct"

outputs = get_completion(prompts, model, temperature=0, top_p=0.9)

# 初始化共享变量
Gastrointestinal_list = []
Hepatobiliary_list = []
Urology_list = []
Cardiovascular_list = []
Thoracic_list = []
Orthopedic_list = []
Neurosurgery_list = []
BurnsandPlastic_list = []
Thyroid_list = []
Breast_list = []
OtherDepartments_list = []
unknown_list = []

unknown_output = []

Gastrointestinal_count = 0
Hepatobiliary_count = 0
Urology_count = 0
Cardiovascular_count = 0
Thoracic_count = 0
Orthopedic_count = 0
Neurosurgery_count = 0
BurnsandPlastic_count = 0
Thyroid_count = 0
Breast_count = 0
OtherDepartments_count = 0
unknown_count = 0

lock = threading.Lock()  # 用于线程同步的锁

# 假设 `outputs` 是已定义的数据
process_outputs(outputs)

ids = 0
# 保存为 .npy 文件
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Gastrointestinal/Gastrointestinal_list_{ids}.npy', Gastrointestinal_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Hepatobiliary/Hepatobiliary_list_{ids}.npy', Hepatobiliary_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Urology/Urology_list_{ids}.npy', Urology_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Cardiovascular/Cardiovascular_list_{ids}.npy', Cardiovascular_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Thoracic/Thoracic_list_{ids}.npy', Thoracic_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Orthopedic/Orthopedic_list_{ids}.npy', Orthopedic_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Neurosurgery/Neurosurgery_list_{ids}.npy', Neurosurgery_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/BurnsandPlastic/BurnsandPlastic_list_{ids}.npy', BurnsandPlastic_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Thyroid/Thyroid_list_{ids}.npy', Thyroid_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/Breast/Breast_list_{ids}.npy', Breast_list)

np.save(f'/home/ubuntu/disk/medical_data/Surgery/OtherDepartments/OtherDepartments_list_{ids}.npy', OtherDepartments_list)
np.save(f'/home/ubuntu/disk/medical_data/Surgery/unknown/unknown_list_{ids}.npy', unknown_list)

# 保存为 JSON 文件
with open(f'/home/ubuntu/disk/medical_data/Surgery/unknown/unknown_output_{ids}.json', 'w') as f:
    json.dump(unknown_output, f)
    

print(f"运行结束, 1 有 {Gastrointestinal_count} 个目标")
print(f"运行结束, 2 有 {Hepatobiliary_count} 个目标")
print(f"运行结束, 3 有 {Urology_count} 个目标")
print(f"运行结束, 4 有 {Cardiovascular_count} 个目标")
print(f"运行结束, 5 有 {Thoracic_count} 个目标")
print(f"运行结束, 6 有 {Orthopedic_count} 个目标")
print(f"运行结束, 7 有 {Neurosurgery_count} 个目标")
print(f"运行结束, 8 有 {BurnsandPlastic_count} 个目标")
print(f"运行结束, 9 有 {Thyroid_count} 个目标")
print(f"运行结束, 10 有 {Breast_count} 个目标")

print(f"运行结束，其他科室有 {OtherDepartments_count} 个目标")
print(f"运行结束，有 {unknown_count} 个目标没有被分类")

