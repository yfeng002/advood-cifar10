#  A Unified Perspective on Adversarial and Out-of-Distribution Detection in the Open World

This repository contains the result in our paper and the CIFAR10 model trained with the SVrandom+ method.

## Prerequisites
* Python 3.7+
* [Prettyatable](https://pypi.org/project/prettytable/)
* PyTorch 1.8+ (for New Test only)

## Result
* Extract the result.tar.bz2 and place in the result folder
* Run `python evaluate.py` to show test results.

## New Test (TODO)
* Extract the model.tar.bz2 and place in the model folder
* Testsets SVHN, Gaussian and Uniform will be automatically downloaded or created.
* To include GTSDB as an OOD testset, download the test subset from (https://benchmark.ini.rub.de/) and extract png images into folder data/GTSDB_Test/. 
* To include LSUN as an OOD testset, download the test subset from (https://github.com/fyu/lsun) and extract png images into folder data/LSUN_Test/.
* Run `python evaluate.py --model/CIFAR10-svrandom+.pt` to produce new results that overwrite exiting files in the result/ folder.
