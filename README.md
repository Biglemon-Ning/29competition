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
Numpy = 1.21.5
```
If you need to construct a new environment by ```Conda```, you can use the following command to create a new virtual environment called npu in your device. 
```
<!-- codna create -n npu python=3.7 
conda activate npu
conda install --yes --file requirements.txt -->
conda env create -f requirements.yaml
```

## Operating System
```
Linux 20.04.1 LTS
```
## Startup
First and foremost, you need to create a virtual environment in your device. 
If you want to train a model, please use the following command.
```
python main.py -s train
```
If you are testing the performance of the existing model, please use the following command.
```
python main.py -s test
```
## Contributor
- **Yi-min Fu**
- **Liang-bo Ning**
- **Han-rui Shi**
- **Rui Liu**
- **Shu-qian Zhou**

## Contact Information
biglemon1123@gmail.com
