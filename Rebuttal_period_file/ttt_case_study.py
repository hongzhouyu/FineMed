import torch
import torch.nn as nn
import faiss
import copy
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import time
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# 1. 加载嵌入模型及构建检索系统

embedding_model_name = "/embedding_model"
embed_model = AutoModel.from_pretrained(embedding_model_name)
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

# 已预先计算并保存 finemed 数据集的嵌入和文本
finemed_embeddings = torch.load("/finemed_embeddings.pt")  # 形状: (N, D)
finemed_texts = torch.load("/finemed_texts.pt")             # 形状: (N, )

# 构建 FAISS 索引（L2 距离）
index = faiss.IndexFlatL2(finemed_embeddings.shape[1])
index.add(finemed_embeddings.numpy())

def get_embedding(text):
    """利用嵌入模型生成文本向量"""
    inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # 使用最后一层隐藏状态均值作为句子表示
        output = embed_model(**inputs).last_hidden_state.mean(dim=1)
    return output.squeeze(0)

# 2. 加载主模型

main_model_name = "/FineMedLM-o1"  # 请根据实际模型名称替换
model = AutoModelForCausalLM.from_pretrained(main_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# 3. Test-Time Training 函数（全参数微调）

def test_time_train_and_answer(model, tokenizer, question, num_updates=1, lr=1e-6):
    """
    对给定问题，先利用检索得到相关文本对模型进行全参数微调，
    然后生成回答，最后恢复模型原始参数。
    
    参数:
      model: 主模型
      tokenizer: 主模型对应的分词器
      question: 用户问题（字符串）
      num_updates: 微调步数
      lr: 微调学习率
    返回:
      answer: 生成的回答（字符串）
    """
    # 复制原始模型参数（全参微调，确保每条样本之间互不干扰）
    original_state_dict = copy.deepcopy(model.state_dict())
    model.train()
    
    ttt_start_time = time.time()

    # 检索最相似的数据
    question_embedding = get_embedding(question).unsqueeze(0)  # 增加 batch 维度
    _, I = index.search(question_embedding.numpy(), 3)  # 检索前 3 个最近邻

    # 从这 3 条中随机选择一条
    selected_index = random.choice(I[0])  # I 是二维数组，I[0] 是 shape=(3,)
    retrieved_text = finemed_texts[selected_index]
    
    # 利用检索到的文本进行全参数微调
    # 构造训练 prompt，此处直接使用检索到的文本作为训练数据
    train_text = retrieved_text
    print(f'retrieved_text: {train_text}')

    inputs = tokenizer(train_text, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=6144).to(model.device)
    
    # 标签设置为输入 id，填充部分设置为 -100（不参与损失计算）
    labels = inputs.input_ids.clone()
    labels[inputs.input_ids == tokenizer.pad_token_id] = -100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(num_updates):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    # 使用微调后的模型生成回答
    model.eval()
    generation_inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    print("-----start generate-----")
    start_time = time.time()
    generated_ids = model.generate(
        generation_inputs.input_ids,
        max_new_tokens=2048,
        eos_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds.")
    print(f"TTT took {end_time - ttt_start_time:.2f} seconds.")

    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(answer)

    # 恢复原始模型参数
    model.load_state_dict(original_state_dict)
    
    return answer

# 4. 对 MMLU-Pro 数据集的每一条数据进行 TTT（全参数微调）

if __name__ == "__main__":
    
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
    messages = [
        {"role": "system", "content": """You are a helpful professional doctor. You need to generate an answer based on the given problem and thoroughly explore the problem through a systematic and long-term thinking process to provide a final and accurate solution. This requires a comprehensive cycle of analysis, summary, exploration, re-evaluation, reflection, backtracking and iteration to form a thoughtful thinking process. Use the background information provided in the text to assist in formulating the answer. Follow these answer guidelines:
    1. Please structure your response into two main sections: **Thought** and **Summarization**.
    2. During the **Thought** phase, think step by step based on the given text content. If the text content is used, it must be expressed.
    3. During the **Summarization** phase, based on the thinking process in the thinking phase, give the final answer to the question.
    Here is the question: """},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(text)

    generated_answer = test_time_train_and_answer(model, tokenizer, text)
