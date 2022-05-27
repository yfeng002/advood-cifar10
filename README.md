#  A Unified Perspective on Adversarial and Out-of-Distribution Detection in the Open World

This repository contains the results and the CIFAR10 model trained with the SVrandom+ method in our AAAI 2022 workshop [paper](AAAI2022_Workshops_W17.pdf):

#### Abstract
Deep neural networks (DNN) have achieved near-human classification capability when testing samples are drawn from their training data distribution. However, numerous research also revealed that the performance of DNN can degrade severely when testing samples are maliciously manipulated or out of distribution (OOD). In response, research in both adversarial defense and OOD detection have become very prevalent independently. This paper investigates the interplay between these two approaches and attempts to unify them to increase the robustness of classification systems in the open world, where inputs could be in-distribution (ID), adversarial, OOD, or adversarial-OOD. We find that existing defensive training methods, adversarial or data augmentation based, trade classification and OOD detection performances for robustness to adversaries. We propose an algebraic transformation based data augmentation technique that reduces DNN’s sensitivity to adversarial attacks. In experiments our method achieved a true positive rate (TPR) of 89.2% for OOD detection and an error of 9.3% for classification in the open world setting. Furthermore, its TPR is higher by 16.7%, and classification error is lower by 2.5 times than existing gradient based adversarial training.


#### Reference
Yeli Feng, Daniel Jun Xian Ng and Arvind Easwaran, "A Unified Perspective on Adversarial and Out-of-Distribution Detection in the Open World", Engineering Dependable and Secure Machine Learning Systems (EDSMLS) - co-located with AAAI, 2022.


#### Acknowledgement
This research was funded in part by MoE, Singapore, Tier-2 grant number MOE2019-T2-2-040.




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

