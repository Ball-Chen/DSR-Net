# Deep Sub-region Network for Salient Object Detection

by Liansheng Wang, Rongzhen Chen, Lei Zhu, Haoran Xie, and Xiaomeng Li.

This implementation is written by Rongzhen Chen at the Xiamen University.

***


## Saliency Map
The results of salienct object detection on five datasets (PASCAL-S, HKU-IS, DUTS-TE, DUT-OMRON and ECSSD) can be found 
at 
Link:https://pan.baidu.com/s/17OHcy6HAYcrNy15IivQEZg  
Password: 3qiv

## Trained Model
You can download the trained model and pretrained DenseNet model which are reported in our paper at 
Link: https://pan.baidu.com/s/1nZ6im4Ero8nhekX-555x-w  
Password: 5qtz

## Requirement
* Python 2.7
* Tensorflow 1.11.0
* Keras 2.2.4
* numpy
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training
1. Set the DUTS-TR dataset as the training set
2. Run by ```python train.py```


*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently 
change them as you need.

Training a model on two GTX 1080Ti GPUs.

## Testing
1. Using PASCAL-S, HKU-IS, DUTS-TE, DUT-OMRON and ECSSD datasets to evaluate our model
2. Run by ```python infer.py```

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently 
change them as you need.
