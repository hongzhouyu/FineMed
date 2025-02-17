import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import seaborn as sns
from datasets import load_from_disk
def get_embeddings(texts, model_name="/root/embedding", batch_size=32, device="cuda"):
    """
    使用预训练模型将文本转换为embeddings
    
    参数:
    texts: 文本列表
    model_name: 模型名称
    batch_size: 批处理大小
    device: 使用的设备 ('cuda' 或 'cpu')
    """
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    
    # 分批处理文本
    for i in tqdm(range(0, len(texts), batch_size), desc="生成embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # 编码文本
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        # 生成embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # 使用[CLS]标记的输出作为句子表示
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    # 合并所有批次的embeddings
    embeddings = np.vstack(embeddings)
    return embeddings

def visualize_tsne_comparison_multiple(datasets, titles, labels=None):
    """
    对多个数据集进行t-SNE可视化比较
    
    参数:
    datasets: list, 每个数据集的embedding列表
    titles: list, 每个数据集的名称
    labels: list, 每个数据集的标签 (可选)
    """
    plt.figure(figsize=(12, 10))
    
    # 合并所有数据集进行 t-SNE 降维
    combined_data = np.vstack(datasets)
    tsne = TSNE(n_components=2, perplexity=25, learning_rate='auto', n_iter=1000)
    combined_tsne = tsne.fit_transform(combined_data)
    
    # 分割 t-SNE 结果并绘制
    start = 0
    for i, data in enumerate(datasets):
        end = start + len(data)
        tsne_result = combined_tsne[start:end]
        plt.scatter(
            tsne_result[:, 0], tsne_result[:, 1],
            alpha=0.3, label=titles[i],
            s=30  # 点的大小
        )
        start = end
    
    plt.title("t-SNE Visualization of Multiple Datasets")
    # plt.xlabel('t-SNE 1') 
    # plt.ylabel('t-SNE 2')
    plt.legend()
    
    # 获取当前时间并格式化
    # from datetime import datetime
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存图片时加上时间戳
    save_path = f"/root/figure/fig_{SEED}.jpg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.show()
    plt.savefig(f"/root/figure/Distribution_{SEED}.pdf", format="pdf")

# 示例使用
if __name__ == "__main__":
    # 加载 5 个数据集
    datasets_paths = [
        "/root/Datasets/3StagesSFTData/InternalMedicine/dataset",
        "/root/Datasets/3StagesSFTData/Surgery/dataset",
        "/root/Datasets/3StagesSFTData/ObstetricsandGynecology/dataset",
        "/root/Datasets/3StagesSFTData/Pediatrics/dataset",
        "/root/Datasets/3StagesSFTData/Otorhinolaryngology/dataset"
    ]
    
    datasets = [load_from_disk(path) for path in datasets_paths]
    # 确定随机种子
    SEED = 4  # 您可以更改为任意整数
    random.seed(SEED)
    np.random.seed(SEED)

    # 每个数据集随机采样 1000 条数据
    sampled_datasets = [ds.select(random.sample(range(len(ds)), 1000)) for ds in datasets]
    
    # 提取文本
    texts_list = [[item['instruction'] for item in dataset] for dataset in sampled_datasets]
    
    # 提取 embeddings
    embeddings_list = []
    for i, texts in enumerate(texts_list):
        print(f"正在生成数据集 {i+1} 的 embeddings...")
        embeddings = get_embeddings(texts)
        embeddings_list.append(embeddings)
    
    # 可视化比较
    visualize_tsne_comparison_multiple(
        datasets=embeddings_list,
        titles=["Internal Medicine", "Surgery", "Obstetrics and Gynecology", "Pediatrics", "Otorhinolaryngology"]
    )
