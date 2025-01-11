from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from datasets import Dataset

import os


def get_completion(prompts, model, tokenizer=None, max_tokens=4096, temperature=0, top_p=0.95):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,  max_tokens=max_tokens)
    llm = LLM(model=model, tokenizer=tokenizer, trust_remote_code=True, tensor_parallel_size=4, max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return outputs


ins_prompt = load_from_disk("data/scoring/ins1_prompt")
# ins_prompt = load_from_disk("data/scoring/ins2_prompt")

print(ins_prompt)
ins_prompt = ins_prompt["prompt"]
# print(prompts[0])
model = "qwen72b"
ins_outputs = get_completion(ins_prompt, model, temperature=0.8, top_p=0.9)


ins_output_list = []
for i, output in enumerate(tqdm(ins_outputs)):
    ins_output_list.append(output.outputs[0].text)


# Save first and then parse to prevent json format errors
# Save as JSON file
with open("data/output_list/scoring_1.json", "w") as file:
    json.dump(ins_output_list, file)

# with open("data/output_list/scoring_2.json", "w") as file:
#     json.dump(ins_output_list, file)

print("Instruction scoring completed!")

# Parse the output json file into dataset class data, see output2datasets.ipynb.
