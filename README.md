
## This document explains the instructions to run the [DeepDIG](https://arxiv.org/pdf/1912.11460.pdf) project


![DeepDIG](http://cse.msu.edu/~karimiha/images/StepWiseDeepDIG.jpg)

## Initialization

1. Clone the project by running `git clone git@github.com:hamidkarimi/DeepDIG.git` in a path on your machine. Let's call this path _CodePath_
2. Run `initial_script.py`. It prompts for a path on your machine to create a directory holding the data, results, etc. Make sure there is enough space in that path where you place the project (at least a couple of GBs). Let's call this path _ProjectPath_
3. Copy or cut **Data** from the cloned Github repository to _ProjectPath_/DeepDIG/

4. MNIST and FASHIONMNIST data are already uploaded. For CIFAR10, download the data from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), unzip it and copy the files into _ProjectPath_/DeepDIG/CIFAR10

4. In `config.py` change  the variable PATH to _ProjectPath_ i.e., the path that you entered when running `initial_script.py`



## Training a pre-trained model

1. Open a terminal and go to the upper-level directory containing the DeepDIG code i.e., where you cloned the code `/cd _CodePath`
2. run `python -m DeepDIG.PreTrainedModels.{DATASET}.{MODEL}.train --dataset {DATASET} --pre-trained-model {MODEL}`
        where `DATASET` is the name of the dataset and `MODEL` is the name of the model
	
	**Example**: `python -m DeepDIGCode.PreTrainedModels.FASHIONMNIST.CNN.train --dataset FASHIONMNIST --pre-trained-model CNN` this will train the _CNN_ model for _FASHIONMNIST_ and then saves it.

Please refer to [here](https://github.com/hamidkarimi/DeepDIG/wiki/Run-DeepDIG-against-a-new-dataset-or-model) to see how you can run DeepDIG against your new dataset/model.

## Running the DeepDIG framework ([Figure 2](https://arxiv.org/pdf/1912.11460.pdf)) 

1. Open a terminal and go to the upper-level directory containing DeepDIG code where you cloned the code `/cd _CodePath`
2. Run `python -m DeepDIGCode.main --dataset {DATASET} --pre-trained-model {MODEL} --classes {s};{t}`
          where `DATASET` is the name of the dataset, `MODEL` is the name of the model, and s and t are two classes in the dataset for which you intend to DeepDIG 

**Example 2:** `python -m DeepDIGCode.main --dataset FASHIONMNIST --pre-trained-model CNN --classes 1;2 `

this will run DeepDIG against the trained _CNN_ on _FASHIONMNIST_ to characterize the decision boundary of classes 1 and 2 (i.e., _Trouser_ and _Pullover_) 

**Note 1.** See here for the explantion of [DeepDIG's arguments](https://github.com/hamidkarimi/DeepDIG/wiki/Arguments-explanation). 

**Note 2.** That model should be trained before as explained above.

**Note 3.** Classes are referred numerically from 0 to n-1 where n is the number of classes. For instance, you can find the classes of CIFAR10 [here](https://www.cs.toronto.edu/~kriz/cifar.html). See the following examples


**Example 2:** `python -m DeepDIGCode.main --dataset CIFAR10 --pre-trained-model GoogleNet --classes 1;2 `
 (_automobile_, _bird_)

**Example 3:** `python -m DeepDIGCode.main --dataset CIFAR10 --pre-trained-model ResNet --classes 4;8`
(_deer_, _ship_)

3.  All results including visualizations will be saved in the _CodePath/DeepDIG/{DATASET}/{PretrainedModel}/{(s,t)}
    where DATASET is the input dataset, PretrainedModel is the given pre-trained model, and s and t are input classes  e.g. _CodePath/DeepDIG/CIFAR10/ResNet/(4,8) in **Example 3**

## Citations

If you use the code in this repository, please cite the following papers

@article{karimi2019characterizing,
  title={Characterizing the Decision Boundary of Deep Neural Networks},
  author={Karimi, Hamid and Derr, Tyler and Tang, Jiliang},
  journal={arXiv preprint arXiv:1912.11460},
  year={2019}
}


@inproceedings{karimi2020decision,
  title={Decision Boundary of Deep Neural Networks: Challenges and Opportunities},
  author={Karimi, Hamid and Tang, Jiliang},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={919--920},
  year={2020}
}

## Contact
Web page: [http://cse.msu.edu/~karimiha/](http://cse.msu.edu/~karimiha/)
Email: [karimiha@msu.edu](karimiha@msu.edu)
