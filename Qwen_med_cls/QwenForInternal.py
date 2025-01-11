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
    global RespiratoryandCritical_count, Cardiology_count, Endocrinology_count, Gastroenterology_count, Hematology_count, Nephrology_count, RheumatologyandImmunology_count, Neurology_count, OtherDepartments_count, unknown_count
    generated_text = output.outputs[0].text
    # print(f"第 {i} 个输出：{generated_text}\n")
    if generated_text == 'Respiratory and Critical Care Medicine':
        with lock:  # 加锁以确保线程安全
            RespiratoryandCritical_list.append(i)
            RespiratoryandCritical_count += 1
    elif generated_text == 'Cardiology':
        with lock:  # 加锁以确保线程安全
            Cardiology_list.append(i)
            Cardiology_count += 1
    elif generated_text == 'Endocrinology':
        with lock:  # 加锁以确保线程安全
            Endocrinology_list.append(i)
            Endocrinology_count += 1
    elif generated_text == 'Gastroenterology':
        with lock:  # 加锁以确保线程安全
            Gastroenterology_list.append(i)
            Gastroenterology_count += 1
    elif generated_text == 'Hematology':
        with lock:  # 加锁以确保线程安全
            Hematology_list.append(i)
            Hematology_count += 1
    elif generated_text == 'Nephrology':
        with lock:  # 加锁以确保线程安全
            Nephrology_list.append(i)
            Nephrology_count += 1
    elif generated_text == 'Rheumatology and Immunology':
        with lock:  # 加锁以确保线程安全
            RheumatologyandImmunology_list.append(i)
            RheumatologyandImmunology_count += 1
    elif generated_text == 'Neurology':
        with lock:  # 加锁以确保线程安全
            Neurology_list.append(i)
            Neurology_count += 1
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
RespiratoryandCritical_list = []
Cardiology_list = []
Endocrinology_list = []
Gastroenterology_list = []
Hematology_list = []
Nephrology_list = []
RheumatologyandImmunology_list = []
Neurology_list = []
OtherDepartments_list = []
unknown_list = []

unknown_output = []

RespiratoryandCritical_count = 0
Cardiology_count = 0
Endocrinology_count = 0
Gastroenterology_count = 0
Hematology_count = 0
Nephrology_count = 0
RheumatologyandImmunology_count = 0
Neurology_count = 0
OtherDepartments_count = 0
unknown_count = 0

lock = threading.Lock()  # 用于线程同步的锁

# 假设 `outputs` 是已定义的数据
process_outputs(outputs)

ids = 0
# 保存为 .npy 文件
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/RespiratoryandCritical/RespiratoryandCritical_list_{ids}.npy', RespiratoryandCritical_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Cardiology/Cardiology_list_{ids}.npy', Cardiology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Endocrinology/Endocrinology_list_{ids}.npy', Endocrinology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Gastroenterology/Gastroenterology_list_{ids}.npy', Gastroenterology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Hematology/Hematology_list_{ids}.npy', Hematology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Nephrology/Nephrology_list_{ids}.npy', Nephrology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/RheumatologyandImmunology/RheumatologyandImmunology_list_{ids}.npy', RheumatologyandImmunology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/Neurology/Neurology_list_{ids}.npy', Neurology_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/OtherDepartments/OtherDepartments_list_{ids}.npy', OtherDepartments_list)
np.save(f'/home/ubuntu/disk/medical_data/InternalMedicine/unknown/unknown_list_{ids}.npy', unknown_list)

# 保存为 JSON 文件
with open(f'/home/ubuntu/disk/medical_data/InternalMedicine/unknown/unknown_output_{ids}.json', 'w') as f:
    json.dump(unknown_output, f)
    
print(f"运行结束, 1 有 {RespiratoryandCritical_count} 个目标")
print(f"运行结束, 2 有 {Cardiology_count} 个目标")
print(f"运行结束, 3 有 {Endocrinology_count} 个目标")
print(f"运行结束, 4 有 {Gastroenterology_count} 个目标")
print(f"运行结束, 5 有 {Hematology_count} 个目标")
print(f"运行结束, 6 有 {Nephrology_count} 个目标")
print(f"运行结束, 7 有 {RheumatologyandImmunology_count} 个目标")
print(f"运行结束, 8 有 {Neurology_count} 个目标")

print(f"运行结束，其他科室有 {OtherDepartments_count} 个目标")
print(f"运行结束，有 {unknown_count} 个目标没有被分类")

