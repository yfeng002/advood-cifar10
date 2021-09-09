#  A Unified Perspective on Adversarial and Out-of-Distribution Detection in the Open World

This repository contains the results from our paper and the CIFAR10 model trained with the SVrandom+ method.

## Prerequisites
* Python 3.7+
* [Prettyatable](https://pypi.org/project/prettytable/)
* PyTorch 1.8+ (for New Test only)
* [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) (for New Test only)


## Result
* Extract the result.tar.bz2 and place in the result folder
* Run `python evaluate.py` to get test results.

## New Test
* Extract the model.tar.bz2 and place in the model folder. 
    * **Tip:** Combine the split model files by running `cat model.tar.bz2.parta* > model.tar.bz2`
* Run `python evaluate.py --model CIFAR10-svrandom+.pt` to produce new results that overwrite exiting files in the result/ folder. **Do not** remove the unzipped results from earlier.
* Testsets CIFAR10, SVHN, Gaussian and Uniform will be automatically downloaded or created. 
* To include GTSDB and LSUN as OOD testsets:
* * Install [Pillow](https://pillow.readthedocs.io/en/stable/)
* * Download the test subset from (https://benchmark.ini.rub.de/) and extract png images into folder data/GTSDB_Test/. 
* * Download the test subset from (https://github.com/fyu/lsun) and extract png images into folder data/LSUN_Test/.

