# README for "Sign-HADiff: Hierarchical Attention Diffusion for Sign Language Generation" Code


[**Projicet Link**](https://doi.org/XXX) 




## 1. Environment Setup 
### 1.1 Dependencies Installation 
```bash
pip install -r requirements.txt
```
## 2. Quick Start

### 2.1 Data Preparation

#### ( 1 ) .Download Dataset 
Download the raw dataset from [[Data  Link](https://pan.baidu.com/s/1JbFhGRRODwMeW57ydENqMQ?pwd=dman)] and place it in /data/phoenix & asl

#### ( 2 ).Download pretrained models
Download the pretrained model from [[Molde Link]](https://pan.baidu.com/s/12praVNueGtFX6wSK_5N5uw?pwd=drvq) and place it in /Models/

### 2.2 Evaluation 
```bash
python __main__.py CVT_test ./config/gsl_config.yaml --ckpt ./Models/gsl.ckpt
python __main__.py CVT_test ./config/asl_config.yaml --ckpt ./Models/asl.ckpt
```


## 3. Contribution & Citation
### 3.1 How to Contribute 

1.Report issues or suggestions via GitHub Issues.

2.Submit pull requests after forking this repository.


## 4. Contact 

Email: yilin_zhang@stu.xidian.edu.cn

Issue Tracker: Submit Issues


