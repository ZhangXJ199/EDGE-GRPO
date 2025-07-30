<h2 align="center">EDGE-GRPO
</a>

<h5 align="center">
<div align="center">

[Xingjian Zhang](https://scholar.google.com/citations?user=H34fwioAAAAJ&hl=zh-CN)<sup>1*</sup>,
[Siwei Wen](https://scholar.google.com/citations?user=kJRiUYwAAAAJ&hl=zh-CN)<sup>1,2*</sup>,
[Wenjun Wu](https://iai.buaa.edu.cn/info/1013/1093.htm)<sup>1,2,3</sup>, 
[Lei Huang](https://huangleibuaa.github.io/)<sup>1,2,3,✉</sup>

<sup>1</sup>SKLCCSE, Institute of Artificial Intelligence, Beihang University, Beijing, China<br>
<sup>2</sup>Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Beihang University, <br>
<sup>3</sup>Hangzhou International Innovation Institute, Beihang University, Hangzhou, China

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2504.09641-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2504.09641)
[![Huggingface](https://img.shields.io/badge/🤗-%20Open%20In%20HF-blue.svg)](https://github.com/ZhangXJ199/EDGE-GRPO)
[![GitHub issues](https://img.shields.io/github/issues/ZhangXJ199/EDGE-GRPO?color=critical&label=Issues)](https://github.com/ZhangXJ199/EDGE-GRPO)
[![GitHub Stars](https://img.shields.io/github/stars/ZhangXJ199/EDGE-GRPO?style=social)](https://github.com/ZhangXJ199/EDGE-GRPO)

</div>

## 📰 News

- [2025-07] Our repository is being completed as soon as possible...
- [2025-07] 🎉 Our arXiv paper [EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction for Advantage Diversity](https://arxiv.org/abs/2507.21848) is released!

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> About

Large Language Models (LLMs) have made remarkable progress in enhancing step-by-step reasoning through reinforcement learning. However, the Group Relative Policy Optimization (GRPO) algorithm, which relies on sparse reward rules, often encounters the issue of identical rewards within groups, leading to the advantage collapse problem. Existing works typically address this challenge from two perspectives: enforcing model reflection to enhance response diversity, and introducing internal feedback to augment the training signal (advantage). In this work, we begin by analyzing the limitations of model reflection and investigating the policy entropy of responses at the fine-grained sample level. Based on our experimental findings, we propose the EDGE-GRPO algorithm, which adopts **E**ntropy-**D**riven Advantage and **G**uided **E**rror Correction to effectively mitigate the problem of advantage collapse. Extensive experiments on several main reasoning benchmarks demonstrate the effectiveness and superiority of our approach.

<div align="center">
<img src="figure/framework.png" alt="framework" width="90%" height="auto">
</div>

## 🛠️ Installation

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/ZhangXJ199/EDGE-GRPO.git
cd EDGE-GRPO
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n edge_grpo python=3.10 -y
conda activate edge_grpo
pip install -r requirements.txt
```

## 📌 Usage

### Trained Model

The model we provided after training: [EDGE-GRPO-Qwen-7B](https://huggingface.co/Zhang199/EDGE-GRPO-Qwen-7B), [EDGE-GRPO-Qwen-1.5B](https://huggingface.co/Zhang199/EDGE-GRPO-Qwen-1.5B)


## 📊 Results

Performance comparison of different methods on three benchmarks during training steps. Our method consistently outperforms the vanilla GRPO and the variant with forced reflection throughout the training process.

<div align="center">
<img src="figure/comparison_during_training.png" alt="framework" width="100%" height="auto">
</div>

Pass@1 performance comparison across various mathematical evaluation benchmarks. The results below are from 1 epoch of training on DeepScaleR-Random-1K. The number of samples in each benchmark is indicated in parentheses. The results are evaluated under the setting of temperature = 0.1. The best results are indicated by **boldface**.

<div align="center">
<img src="figure/main_result.jpg" alt="framework" width="90%" height="auto">
</div>

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/3557/3557963.png">Changes in Entropy and Advantage Variance During Training

<div align="center">
<img src="figure/changes.png" alt="framework" width="100%" height="auto">
</div>

## 📝 Citation



## 📨 Contact

If you have any questions or suggestions, please feel free to contact us at ``zhangxingjian@buaa.edu.cn``.

## ❤️ Community efforts

* This repository is based on [trl](https://github.com/huggingface/trl) project.
* The implementation of evaluation refers to the [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) project. Great work!
