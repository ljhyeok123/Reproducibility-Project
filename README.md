# Reproducibility-Project

# APoT Quantization Paper Reproducibility

This project reproduces APoT Quantization and uses pytorch to show results on CIFAR/ImageNet for various ResNet models. The goal of the project is to make the accuracy drop due to quantization as similar as possible to the paper results.

# APoT Quantization
Additive powers-of-two quantization: An efficient non-uniform discretization for neural networks (APoT) is a paper accepted at ICLR 2020. The paper has 285 citations as of June 3, 2024.

**APoT Quantization논문에서의 양자화 기법 간략하게 추가하기**

# Code Configuration
- `resnet.py`: This code declares the structure of resnet for imageNet/CIFAR and exists in the CIFAR/ImageNet folder, respectively. The reason for this is that the structures of resnet used in imageNet and CIFAR are different. For example, in imageNet, the kernel size of the **first convolution is 7**, while in cifar, the kernel size of the **first convolution is 3**.
- `quant_layer.py`: This code performs **activation/weight quantization and convolution**.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.

# APoT Quantization Method Reproducibility for CIFAR 10(논문 결과 구현하는 부분)

## How to run it 
```
### code download
git clone https://github.com/SangbeomJeong/Reproducibility-Project.git

### run 
python main.py --arch res20 --bit 32 -id 0,1 --wd 5e-4
python main.py --arch res20 --bit 4 -id 0,1 --wd 1e-4  --lr 4e-2 --init result/res20_32bit/model_best.pth.tar
python main.py --arch res20 --bit 3 -id 0,1 --wd 1e-4  --lr 4e-2 --init result/res20_4bit/model_best.pth.tar
python main.py --arch res20 --bit 2 -id 0,1 --wd 3e-5  --lr 4e-2 --init result/res20_3bit/model_best.pth.tar
```
- `-id 0,1`: It uses argparse to specify which GPU to use, and is the same function as CUDA_VISIBLE_DEVICES=0,1.
- `--init result/res20_32bit/model_best.pth.tar`: This is a function that specifies starting from a specific weight. In this example, it means that quantization will be performed from the weight learned with full precision for the resnet20 architecture.

## environment
epoch : 300
optimizer : SGD
lr scheduler : MultiStepLR
batch size : 128


## Performance Table and Analysis
### paper table
#### ResNet20

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    91.6                  |             -          | 
|         4bit           |    92.3                  |         -0.7           | 
|         3bit           |    92.2                  |         -0.6           | 
|         2bit           |    91.0                  |          0.6           | 

#### ResNet56

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    93.2                  |             -          | 
|         4bit           |    94.0                  |         -1.8          | 
|         3bit           |    93.9                 |         -0.7           | 
|         2bit           |    92.9                 |          0.3           | 


### Reproducibility table(Ours)

#### ResNet20

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    92.90                  |             -          | 
|         4bit           |    92.56                  |         0.34          | 
|         3bit           |    92.53                  |         0.37           | 
|         2bit           |    91.02                  |          1.88          | 

#### ResNet56

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    93.94                  |             -          | 
|         4bit           |    93.85                  |         0.09          | 
|         3bit           |    93.46                 |         0.48           | 
|         2bit           |    92.75                 |          1.19           | 


Because we cannot match the completely identical environment (ex: GPU, pytorch version, CUDA version) as the paper, we want to judge the performance using the accuracy drop index in the experimental results. In the actual paper, results exceeded the baseline in all other conditions except for 2bit. On the other hand, our reproducibility results showed results that were lower than baseline in all conditions. Initially, quantization experiments were conducted using pretrained weights provided by the authors of the paper, but performance was lower in all conditions. So, weㄲ conducted an experiment by relearning the baseline from the beginning and securing pretrained weights. **The results of the above reproducibility table (Ours) are the results of quantization experiments obtained directly from the baseline.**


# APoT Quantization Method Reproducibility for ImageNet(논문 결과 구현하는 부분)

Due to GPU performance and memory constraints, only 5-bit experiments were reproduced for ResNet18.

## How to run it 
```
### code download
git clone https://github.com/SangbeomJeong/LeNet5_MLP_project.git    dddddddddd 수정해야함 

### dataset prepare
please prepare the ImageNet validation and training dataset

### run 
python main.py -a resnet18 --bit 5
```


## environment
epoch : 120
optimizer : SGD
lr scheduler : MultiStepLR
batch size : 256

## Performance Table and Analysis
### paper table
#### ResNet18

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |       70.2               |             -          | 
|         5bit           |    70.9                  |         -0.7           | 


### Reproducibility table(Ours)

#### ResNet18

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    70.2                  |             -          | 
|         5bit           |       채워야함                   |  채워야 함       | 


# Ablation & hyperparameter tuning 

We also conducted experiments on CIFAR100 to check whether the APoT quantization method is effective on other datasets. Additionally, in this paper, MultiStepLR is used to adaptively apply the learning rate, but in an attempt to increase performance, we plan to replace it with CosineAnnealingLR and check the results.

## ResNet models on CIFAR 100
In the paper, experiments are conducted only on cifar10, and the results on cifar100 are omitted. We would like to conduct an experiment on cifar100 to check whether the quantization method proposed in the paper can be used universally.

A quantization experiment was conducted based on pretrained weights obtained through scratch learning in the same environment as cifar10 performed above.

#### ResNet20

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    69.09                 |             -          | 
|         4bit           |    68.56                  |         0.53          | 
|         3bit           |    67.56                  |         1.53           | 
|         2bit           |    63.92                  |          5.17          | 

#### ResNet56

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    72.92                  |             -          | 
|         4bit           |    72.19                  |         0.73          | 
|         3bit           |    71.47                 |         1.45           | 
|         2bit           |    69.64                 |          3.28          | 

Looking at the experimental results for CIFAR 100, 4-bit quantization showed a somewhat acceptable accuracy drop for the ResNet20/56 model, but 3/2-bit quantization showed a very large accuracy drop.

From this, we conclude that the APoT method is not suitable for ultra-low-precision quantization techniques such as 3/2bit for cifar100.

## ResNet models on CIFAR10 with CosineAnnealingLR

In an attempt to further improve the previously reproduced performance, we will change the lr scheduler to CosineAnnealingLR and check the results. All environments except the lr scheduler are the same as the existing CIFAR 10 reproducibility environment.

#### ResNet20

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |                    |             -          | 
|         4bit           |                   |                | 
|         3bit           |                   |                | 
|         2bit           |                |                | 


# Conclusion

추가해야함 




