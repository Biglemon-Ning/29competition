# README
This file is used to record the details of this work.

## The functions of different files
In this section, we will introduce the function of each file in this project. 
- ```data_loader.py``` is used to load data from the existing text files and preprocess the data preliminarily.
- ```npu_model.py``` is designed to build a model.
- ```main.py``` is employed to train the model, save the parameters, and test the performance of the existing model.
- ```result_model.pth``` is the saved model that achieves the best perfoemance based on the evaluation dataset.
- ```requirements.txt``` is to record the information of the virtual environment.
- ```output.txt``` is to record the result when testing the performance of model. After the validation process is finished, this file is created.

## Conda Environment
The details are introduced in the ```requirements.txt```. The information of the main packages is shown as follow:
```
Python = 3.7
Pytorch = 1.8.0
Numpy = 1.21
```
If you need to construct a new environment by ```Conda```, you can use the following command to create a new virtual environment called npu in your device. 
First, in order to install package rapidly, you may need to use the Tsinghua Mirror Source.
```
conda config --show channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```
Then you can use the following command to create a new environment called ```npu``` and activate it.
```
conda create -n npu python=3.7 
conda activate npu
```
After that, you should install pytorch in this environment.
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly 
```
If you are puzzled by the vision of any package, you can check out the corresponding information in ```requirements.txt```.

## Operating System
```
Linux 20.04.1 LTS
```
## Startup
First and foremost, you need to create a virtual environment in your device. 
If you want to train a model in ```Windows```, please use the following command.
```
python -X utf8 main.py -s train
```
If you are testing the performance of the existing model in ```Windows```, please use the following command.
```
python -X utf8 main.py -s test
```
## Contributor
- **Yi-min Fu**
- **Liang-bo Ning**
- **Han-rui Shi**
- **Rui Liu**
- **Shu-qian Zhou**

## Contact Information
biglemon1123@gmail.com
