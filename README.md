
## This document explains the instructions to run the [DeepDIG] (https://ieeexplore.ieee.org/abstract/document/10069079)


![DeepDIG](http://cse.msu.edu/~karimiha/images/StepWiseDeepDIG.jpg)

## Initialization

1. Download the zip file of the code from this repository. Unzip it and rename the directory to **DeepDIGCode**. Let's assume this directory in /home/user/Downloads/
2.  In `config.py` change  the variable PATH to /home/user/Downloads/DeepDIGCode/
3. Data for MNIST and FASHIONMNIST are already uploaded. For CIFAR10, download the data from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), unzip it and copy the files into /home/user/Downloads/DeepDIGCode/Data/CIFAR10


## Training the base model

1. Open a terminal and go to the upper-level directory containing the DeepDIG code where you cloned the code e.g., /home/user/Downloads/
2. run `python -m DeepDIGCode.PreTrainedModels.{DATASET}.{MODEL}.train --dataset {DATASET} --pre-trained-model {MODEL}`
        where `DATASET` is the name of the dataset and `MODEL` is the name of the model
	
	**Example**: `python -m DeepDIGCode.PreTrainedModels.FASHIONMNIST.CNN.train --dataset FASHIONMNIST --pre-trained-model CNN` this will train the _CNN_ model for _FASHIONMNIST_ and then saves it.

Please refer to [here](https://github.com/hamidkarimi/DeepDIG/wiki/Run-DeepDIG-against-a-new-dataset-or-model) to see how you can run DeepDIG against your new dataset/model.

## Running the DeepDIG framework ([Figure 2](https://arxiv.org/pdf/1912.11460.pdf)) 

1. Open a terminal and go to the upper-level directory containing DeepDIG code where you cloned the code e.g., /home/user/Downloads/
2. Run `python -m DeepDIGCode.main --dataset {DATASET} --pre-trained-model {MODEL} --classes {s};{t}`
          where `DATASET` is the name of the dataset, `MODEL` is the name of the model, and s and t are two classes in the dataset for which you intend to DeepDIG 

**Example :** `python -m DeepDIGCode.main --dataset FASHIONMNIST --pre-trained-model CNN --classes "1;2" `

this will run DeepDIG against the trained _CNN_ on _FASHIONMNIST_ to characterize the decision boundary of classes 1 and 2 (i.e., _Trouser_ and _Pullover_) 

**Note 1.** See here for the explantion of [DeepDIG's arguments](https://github.com/hamidkarimi/DeepDIG/wiki/Arguments-explanation). 

**Note 2.** Classes are referred numerically from 0 to n-1 where n is the number of classes. For instance, you can find the classes of CIFAR10 [here](https://www.cs.toronto.edu/~kriz/cifar.html). See the following examples


**Example :** `python -m DeepDIGCode.main --dataset CIFAR10 --pre-trained-model GoogleNet --classes "1;2" `
 (_automobile_, _bird_)

**Example :** `python -m DeepDIGCode.main --dataset CIFAR10 --pre-trained-model ResNet --classes "4;8"`
(_deer_, _ship_)

3.  All results including visualizations will be saved in the /home/user/Downloads/DeepDIGCode/PreTrainedModels/{DATASET}/{MODEL}/{(s,t)}
    where DATASET is the input dataset, MODEL is the base model, and s and t are input classes for which you intend to genderate the borderline examples

 e.g. /home/user/Downloads/DeepDIGCode/PreTrainedModels/FASHIONMNIST/CNN/(1,2) 
    

## Citation

If you use the code in this repository, please cite the following paper

@INPROCEEDINGS{Karimi2022DeepDIG,
  author={Karimi, Hamid and Derr, Tyler},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Decision Boundaries of Deep Neural Networks}, 
  year={2022},
  volume={},
  number={},
  pages={1085-1092},
  doi={10.1109/ICMLA55696.2022.00179}
  }


## Contact
Web page: [www.hamidkarimi.com](http://www.hamidkarimi.com)
Email: [hamid.karimi@usu.edu](hamid.karimi@usu.edu)
