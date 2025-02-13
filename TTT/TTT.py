import torch
import torch.nn as nn
import faiss
import copy
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# 1. 加载嵌入模型及构建检索系统

embedding_model_name = "BAAI/bge-large-en-v1.5"
embed_model = AutoModel.from_pretrained(embedding_model_name)
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

# 已预先计算并保存 finemed 数据集的嵌入和文本
finemed_embeddings = torch.load("finemed_embeddings.pt")  # 形状: (N, D)
finemed_texts = torch.load("finemed_texts.pt")             # 形状: (N, )

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

main_model_name = "FineMedLM-o1"  # 请根据实际模型名称替换
model = AutoModelForCausalLM.from_pretrained(main_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# 3. Test-Time Training 函数（全参数微调）

def test_time_train_and_answer(model, tokenizer, question, num_updates=1, lr=1e-5):
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
    
  
    # 检索最相似的数据
    question_embedding = get_embedding(question).unsqueeze(0)  # 增加 batch 维度
    _, I = index.search(question_embedding.numpy(), 1)  # 检索1个最近邻
    retrieved_text = finemed_texts[I[0][0]]
    
    # 利用检索到的文本进行全参数微调
    # 构造训练 prompt，此处直接使用检索到的文本作为训练数据
    train_text = retrieved_text
    inputs = tokenizer(train_text, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=tokenizer.model_max_length).to(model.device)
    
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
    generated_ids = model.generate(
        **generation_inputs,
        max_length=2048,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 恢复原始模型参数
    model.load_state_dict(original_state_dict)
    
    return answer

# 4. 对 MMLU-Pro 数据集的每一条数据进行 TTT（全参数微调）

if __name__ == "__main__":
    # MMLU-Pro 数据集以 JSON 格式保存，每条记录包含 "question" 和可选的 "answer" 字段
    with open("mmlu_pro_data.json", "r", encoding="utf-8") as f:
        mmlu_data = json.load(f)
    
    results = []
    for idx, sample in enumerate(mmlu_data):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")
        
        print(f"正在处理第 {idx + 1} 条样本...")
        print("问题:", question)
        # 对每个问题执行 TTT 得到回答
        generated_answer = test_time_train_and_answer(model, tokenizer, question)
        print("生成的回答:", generated_answer)
        if ground_truth:
            print("正确答案:", ground_truth)
        print("-" * 50)
        
        # 保存结果到列表中，便于后续评估或存盘
        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth
        })
    
    # 将结果保存到文件
    with open("mmlupro_ttt_results.json", "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=4)

