from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import os
import json
from datasets import Dataset


def get_completion(prompts, model, tokenizer=None, max_tokens=4096, temperature=0, top_p=0.95):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,  max_tokens=max_tokens)
    llm = LLM(model=model, tokenizer=tokenizer, trust_remote_code=True, tensor_parallel_size=4, max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return outputs

# common instructions
processed_dataset = load_from_disk("data/responses/res_gen_prompt_common")

# complex instructions
# processed_dataset = load_from_diskdata/responses/res_gen_prompt_complex")


prompts = processed_dataset["prompt"]
# print(prompts[0])

# qwen72b
model = "qwen72b"

# qwq
# model = "qwq/Qwen/QwQ-32B-Preview"


outputs = get_completion(prompts, model, temperature=1, top_p=0.9)

output_list = []
for i, output in enumerate(tqdm(outputs)):
    output_list.append(output.outputs[0].text)

# Save first and then parse to prevent json format errors
# Save as JSON file
with open("data/output_list/res_output_list_common.json", "w") as file:
    json.dump(output_list, file)

# with open("data/output_list/res_output_list_complex1.json", "w") as file:
#     json.dump(output_list, file)
# with open("data/output_list/res_output_list_complex2.json", "w") as file:
#     json.dump(output_list, file)

print("Generating answers completed!")

# Parse the output json file into dataset class data, see output2datasets.ipynb.
