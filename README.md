# What is this repo?

This repo includes the codebase and some introductions of FineMed, as described in the paper [FineMedLM-o1: Enhancing Medical Knowledge Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training](https://arxiv.org/abs/2501.09213).

## Dataset

We have uploaded the datasets to huggingface: 

SFT Datasets: https://huggingface.co/datasets/hongzhouyu/FineMed-SFT

DPO Dataset: https://huggingface.co/datasets/hongzhouyu/FineMed-DPO

## Models

We have uploaded the models to huggingface:

FineMedLM: https://huggingface.co/hongzhouyu/FineMedLM

FineMedLM-o1: https://huggingface.co/hongzhouyu/FineMedLM-o1

## Reproduction

If you want to reproduce our research, please run the following code in sequence:

1. Synthetic Data

2. Qwen_med_cls

3. Training

4. TTT

## Email

If you are interested in FineMed, feel free to email me at hzyu24@m.fudan.edu.cn!

## Citing FineMed:

If FineMed or this repository is useful in your own research, you can use the following BibTeX entry:

    @misc{yu2025finemedlmo1enhancingmedicalknowledge,
      title={FineMedLM-o1: Enhancing Medical Knowledge Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training}, 
      author={Hongzhou Yu and Tianhao Cheng and Yingwen Wang and Wen He and Qing Wang and Ying Cheng and Yuejie Zhang and Rui Feng and Xiaobo Zhang},
      year={2025},
      eprint={2501.09213},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.09213}, 
    }
