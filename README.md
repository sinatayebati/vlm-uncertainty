<div align='center'>

<p align="center">
    <img src="https://github.com/sinatayebati/vlm-uncertainty/blob/gh-pages/static/images/logo.webp" width="100" style="margin-bottom: 0.2;"/>
</p>

# Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models

<h5 align="center"> Please give us a star ⭐ if you find this work useful  </h5>

<h5 align="center">

 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.06607-b31b1b.svg?logo=arXiv)]() 
[![Website](https://img.shields.io/badge/Website-Link-darkgreen)](https://sinatayebati.github.io/vlm-uncertainty/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/sinatayebati/vlm-uncertainty/blob/main/LICENSE) 
 <br>
</h5>

</div>


## News 
* ```2025.2.06``` 🚀 We release the paper [arXiv](https://arxiv.org/).

## 🐳 Model Zoo

### Vision-Language Models (VLMs) 🖼️

| Model Series | Model Name | Parameters | Architecture |
|--------------|------------|------------|--------------|
| LLaVA | LLaVA-v1.6-34B | 34B | Vision-Language |
| | LLaVA-v1.6-13B | 13B | Vision-Language |
| | LLaVA-v1.6-7B | 7B | Vision-Language |
| Lightweight | MoE-LLaVA-Phi2 | 2.7B | Vision-Language |
| | MobileVLM-v2 | 7B | Vision-Language |
| Other VLMs | mPLUG-Owl2 | 7B | Vision-Language |
| | Qwen-VL-Chat | 7B | Vision-Language |
| | Yi-VL | 6B | Vision-Language |
| | CogAgent-VQA | 7B | Vision-Language |

### Large Language Models (LLMs) 📚

| Model Series | Model Name | Parameters | Architecture |
|--------------|------------|------------|--------------|
| Yi | Yi-34B | 34B | Language |
| Qwen | Qwen-14B | 14B | Language |
| | Qwen-7B | 7B | Language |
| Llama-2 | Llama-2-13B | 13B | Language |
| | Llama-2-7B | 7B | Language |

## 📊 Evaluation
### Key Improvements Over Baselines 🚀

- **Hallucination Detection**: Up to 22.19% improvement in AUROC
- **Uncertainty Estimation**: 21.17% boost in uncertainty-guided selective generation (AUARC)
- **Calibration**: 70-85% reduction in calibration error
- **Coverage**: Consistently meets 90% coverage target while reducing prediction set size

### Benchmarks 🔖
- for detailed results, please refer to the [paper](https://arxiv.org/).


## 🤖 Getting started

6 groups of models could be launch from one environment: LLaVa, CogVLM, Yi-VL, Qwen-VL,
internlm-xcomposer, MoE-LLaVA. This environment could be created by the following code:
```shell
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/haotian-liu/LLaVA.git 
pip install git+https://github.com/PKU-YuanGroup/MoE-LLaVA.git --no-deps
pip install deepspeed==0.9.5
pip install -r requirements.txt
pip install xformers==0.0.23 --no-deps
```
mPLUG-Owl model can be launched from the following environment:
```shell
python3 -m venv venv_mplug
source venv_mplug/bin/activate
git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG-Owl/mPLUG-Owl2
git checkout 74f6be9f0b8d42f4c0ff9142a405481e0f859e5c
pip install -e .
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
cd ../../
pip install -r requirements.txt
```
Monkey models can be launched from the following environment:
```shell
python3 -m venv venv_monkey
source venv_monkey/bin/activate
git clone https://github.com/Yuliang-Liu/Monkey.git
cd ./Monkey
pip install -r requirements.txt
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
cd ../
pip install -r requirements.txt
```

To check all models you can run ```scripts/test_model_logits.sh```


To work with Yi-VL:
```shell
apt-get install git-lfs
cd ../
git clone https://huggingface.co/01-ai/Yi-VL-6B
```


### Model logits

To get model logits in four benchmarks run command from `scripts/run.sh`.

### To train the abstention model with RL
```shell
bash scripts/train_all_models.sh
```

### To evaluate the abstention model + uncertainty quantification benchmark
```shell
bash scripts/evaluate_policies.sh
```