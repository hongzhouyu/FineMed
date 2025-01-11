from transformers import AutoTokenizer, pipeline
import torch
from datasets import load_from_disk
import numpy as np
import random
from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator
import time

# accelerate config 
# accelerate launch 


messages_template = [
    {"role": "user", "content": "{instruction}"},
    {"role": "assistant", "content": "{response}"}
]

rm_tokenizer = AutoTokenizer.from_pretrained("RM")


def create_messages(instruction, response):
    return [
        {"role": "user", "content": instruction} if msg["role"] == "user" 
        else {"role": "assistant", "content": response}
        for msg in messages_template
    ]

def create_prompt1(example):
    message = create_messages(example["instruction"], example["response1"])
    text = rm_tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False
    ).replace(rm_tokenizer.bos_token, "")
    example["prompt"] = text
    return example

def create_prompt2(example):
    message = create_messages(example["instruction"], example["response2"])
    text = rm_tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False
    ).replace(rm_tokenizer.bos_token, "")
    example["prompt"] = text
    return example

def AddTypeCommon(example):
    example["instruction_type"] = "common"
    return example
def AddTypeComplex(example):
    example["instruction_type"] = "complex"
    return example

common_ins = load_from_disk("sft_dataset/o1")
# complex_ins = load_from_disk("data1/responses/complex_ins")
print("Before filtering with successful_indexes:")
print(common_ins)

# Loading .npy files
successful_indexes = np.load('sft_dataset/successful_indexes.npy')
print(len(successful_indexes))
common_ins = common_ins.select(successful_indexes)
print("After filtering with successful_indexes:")
print(common_ins)

common_answer1_list = load_from_disk("sft_dataset/res1")["answer"]
common_answer2_list = load_from_disk("sft_dataset/res2")["answer"]
common_ins = common_ins.add_column("response1", common_answer1_list)
common_ins = common_ins.add_column("response2", common_answer2_list)
print("Before screening with RM -> common:")
print(common_ins)

# complex_answer1_list = load_from_disk("data1/responses/gen_res/res1_qwq")["answer"]
# complex_answer2_list = load_from_disk("data1/responses/gen_res/res2_qwq")["answer"]
# complex_ins = complex_ins.add_column("response1", complex_answer1_list)
# complex_ins = complex_ins.add_column("response2", complex_answer2_list)
# print("Before screening with RM -> complex:")
# print(complex_ins)

common_ins1 = common_ins.map(create_prompt1, remove_columns=common_ins.column_names, num_proc=128)
# complex_ins1 = complex_ins.map(create_prompt1, remove_columns=complex_ins.column_names, num_proc=128)

common_ins2 = common_ins.map(create_prompt2, remove_columns=common_ins.column_names, num_proc=128)
# complex_ins2 = complex_ins.map(create_prompt2, remove_columns=complex_ins.column_names, num_proc=128)

# Using Accelerator to manage devices
accelerator = Accelerator()

# Create pipeline
rm_pipe = pipeline(
    "sentiment-analysis",
    model="RM",
    tokenizer=rm_tokenizer,
    device=accelerator.device,  
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 64
}

print("start scoring common response1")
start_time = time.time()  
pipe_outputs1 = rm_pipe(common_ins1["prompt"], **pipe_kwargs)
common_rewards1 = [output[0]["score"] for output in pipe_outputs1]
end_time = time.time()
execution_time = end_time - start_time
print(f"Program execution time: {execution_time:.2f} seconds")
print(common_rewards1[0], len(common_rewards1))


print("start scoring common response2")
start_time = time.time()  
pipe_outputs2 = rm_pipe(common_ins2["prompt"], **pipe_kwargs)
common_rewards2 = [output[0]["score"] for output in pipe_outputs2]
end_time = time.time()  
execution_time = end_time - start_time
print(f"Program execution time: {execution_time:.2f} seconds")
print(common_rewards2[0], len(common_rewards2))

# print("start scoring complex responses")
# start_time = time.time()  
# pipe_outputs1 = rm_pipe(complex_ins1["prompt"], **pipe_kwargs)
# complex_rewards1 = [output[0]["score"] for output in pipe_outputs1]
# pipe_outputs2 = rm_pipe(complex_ins2["prompt"], **pipe_kwargs)
# complex_rewards2 = [output[0]["score"] for output in pipe_outputs2]
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Program execution time: {execution_time:.2f} seconds")
# print(complex_rewards1[0], complex_rewards2[0], len(complex_rewards1), len(complex_rewards2))

# Comparing RM scores
print("start comparing RM score of common res:")
common_res_list = []
for r1, r2 in zip(common_rewards1, common_rewards2):
    if r1 > r2:
        common_res_list.append(1)
    elif r1 < r2:
        common_res_list.append(2)
    else:
        common_res_list.append(random.choice([1, 2]))
print(common_res_list[:5], len(common_res_list))

# print("start comparing RM score of complex res:")
# complex_res_list = []
# for r1, r2 in zip(complex_rewards1, complex_rewards2):
#     if r1 > r2:
#         complex_res_list.append(1)
#     elif r1 < r2:
#         complex_res_list.append(2)
#     else:
#         complex_res_list.append(random.choice([1, 2]))
# print(complex_res_list[:5], len(complex_res_list))

print("start selecting common res:")
common_answer_list = []
for ids, v in enumerate(common_res_list):
    if v == 1:
        common_answer_list.append(common_ins[ids]['response1'])
    else:
        common_answer_list.append(common_ins[ids]['response2'])
print(len(common_answer_list))

# print("start selecting complex res:")
# complex_answer_list = []
# for ids, v in enumerate(complex_res_list):
#     if v == 1:
#         complex_answer_list.append(complex_ins[ids]['response1'])
#     else:
#         complex_answer_list.append(complex_ins[ids]['response2'])
# print(len(complex_answer_list))

common_ins = common_ins.add_column("common_response", common_answer_list)
# common_ins = common_ins.remove_columns(['ids', 'response1', 'response2'])


# complex_ins = complex_ins.add_column("response", complex_answer_list)
# complex_ins = complex_ins.remove_columns(['ids', 'response1', 'response2'])


common_ins = common_ins.map(AddTypeCommon)
# complex_ins = complex_ins.map(AddTypeComplex)

# Merge two datasets
ins_res = concatenate_datasets([common_ins, complex_ins])

# Shuffle the dataset
ins_res = ins_res.shuffle(seed=42)
print("final SFT dataset:")
print(ins_res)
print(ins_res[0])

ins_res.save_to_disk("sft_dataset/o1_common")
print("RM filter ending!")
