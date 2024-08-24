# UniGen: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation

## Introduction

This repository contains the source code for the paper ["UniGen: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation"](https://arxiv.org/pdf/2405.01022). We propose a novel domain generalization method, UniGen, which is based on zero-shot dataset generation. UniGen generates a new synthetic dataset without a specified target domain, thus enabling the model to generalize to unseen domains. Please refer to the paper for more details.

![Figure](./unigen_figure.jpg)

## Experiment

```shell
$ conda create -n proj-unigen python=3.8
$ conda activate proj-unigen
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
$ bash run_experiment.sh
```

## Citation

```bibtex
@article{choi2024unigen,
  title={UniGen: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation},
  author={Choi, Juhwan and Kim, Yeonghwa and Yu, Seunguk and Yun, JungMin and Kim, YoungBin},
  journal={arXiv preprint arXiv:2405.01022},
  year={2024}
}
```
