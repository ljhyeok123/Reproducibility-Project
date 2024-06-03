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
git clone https://github.com/SangbeomJeong/LeNet5_MLP_project.git

### run 
python main.py --arch res20 --bit 32 -id 0,1 --wd 5e-4
python main.py --arch res20 --bit 4 -id 0,1 --wd 1e-4  --lr 4e-2 --init result/res20_32bit/model_best.pth.tar
python main.py --arch res20 --bit 3 -id 0,1 --wd 1e-4  --lr 4e-2 --init result/res20_4bit/model_best.pth.tar
python main.py --arch res20 --bit 2 -id 0,1 --wd 3e-5  --lr 4e-2 --init result/res20_3bit/model_best.pth.tar
```
- `-id 0,1`: It uses argparse to specify which GPU to use, and is the same function as CUDA_VISIBLE_DEVICES=0,1.
- `--init result/res20_32bit/model_best.pth.tar`: This is a function that specifies starting from a specific weight. In this example, it means that quantization will be performed from the weight learned with full precision for the resnet20 architecture.

## Performance Summary 
### paper table
#### ResNet20

|   Architecture         |      Accuracy(%)         |      Accuracy drop(%)  | 
|------------------------|--------------------------|------------------------|
|     Basline(FP32)      |    91.6                  |         -
|         4bit           |    92.3                  |         -0.7           | 
|         3bit           |    92.2                  |         -0.6           | 
|         2bit           |    91.0                  |          0.6           | 








### Reproducibility table(Ours)
