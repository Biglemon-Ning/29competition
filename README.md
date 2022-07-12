# README
This file is used to record the details of this work.

## Method
This section is used to present the details of this work

## The functions of different files
In this section, we will introduce the function of each file in this project. 
- ```data_loader.py``` is used to load data from the existing text files and preprocess the data preliminarily.
- ```npu_model.py``` is designed to build a model.
- ```train.py``` is employed to train the model and save the parameters.
- ```test.py``` is to test the effectiveness of the existing model. 
- ```result_model.pth``` is the model that achieve the best perfoemance based on the evaluation dataset.

## Conda Environment
The details are introduced in the ```requirements.txt```. The information of the main package is shown as follow:
```
Python = 3.7
Pytorch = 1.8.0
Numpy = 1.21.5
```
If you need to construct a new environment by ```Conda```, you can use the following command in your device.
```
conda install --yes --file requirements.txt
```

## Operating System
```
Linux 20.04.1 LTS
```
## Startup
If you want to train a model, please use the following command.
```
python train.py
```
If you are testing the performance of the existing model, please use the following command.
```
python test.py
```
## Contributor
**Liang-bo Ning**

**Yi-min Fu**

## Contact Information
biglemon1123@gmail.com
