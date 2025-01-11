from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import os
import json


def get_completion(prompts, model, tokenizer=None, max_tokens=4096, temperature=0, top_p=0.95):

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,  max_tokens=max_tokens)
    # Initialize the vLLM inference engine
    llm = LLM(model=model, tokenizer=tokenizer, trust_remote_code=True, tensor_parallel_size=4, max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return outputs

processed_dataset = load_from_disk("path")

prompts = processed_dataset["prompt"]
# print(prompts[0])
model = "path/to/qwen72b"
outputs = get_completion(prompts, model, temperature=1, top_p=0.9)

output_list = []
for i, output in enumerate(tqdm(outputs)):
    output_list.append(output.outputs[0].text)


# Save first and then parse to prevent json format errors
# Save as JSON file
with open("path/to/output_list/ins_gen_output_list.json", "w") as file:
    json.dump(output_list, file)

print("Generate instructions complete!")

# Parse the output json file into dataset class data, see output2datasets.ipynb.
