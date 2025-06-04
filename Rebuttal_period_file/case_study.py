from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# 指定模型路径
main_model_name = "/FineMedLM"  # 请根据实际模型名称替换
model = AutoModelForCausalLM.from_pretrained(main_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# 构造输入
prompt = (
    """The following are multiple choice questions (with answers) about health. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.


Question:
Polio can be eradicated by which of the following?
Options:
A. Herbal remedies
B. Use of antibiotics
C. Regular intake of vitamins
D. Administration of tetanus vaccine
E. Attention to sewage control and hygiene
F. Natural immunity acquired through exposure
G. Use of antiviral drugs
Answer: Let's think step by step.
"""
)

# FineMedLM
messages = [
    {"role": "system", "content": "You are a helpful professional doctor. The user will give you a medical question, and you should answer it in a professional way."},
    {"role": "user", "content": prompt}
]

# FineMedLM-o1
# messages = [
#     {"role": "system", "content": """You are a helpful professional doctor. You need to generate an answer based on the given problem and thoroughly explore the problem through a systematic and long-term thinking process to provide a final and accurate solution. This requires a comprehensive cycle of analysis, summary, exploration, re-evaluation, reflection, backtracking and iteration to form a thoughtful thinking process. Use the background information provided in the text to assist in formulating the answer. Follow these answer guidelines:
# 1. Please structure your response into two main sections: **Thought** and **Summarization**.
# 2. During the **Thought** phase, think step by step based on the given text content. If the text content is used, it must be expressed.
# 3. During the **Summarization** phase, based on the thinking process in the thinking phase, give the final answer to the question.
# Here is the question: """},
#     {"role": "user", "content": prompt}
# ]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 推理并计时
print("-----start generate-----")
start_time = time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
)
end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds.")

# 解码输出

answer = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print(answer)
