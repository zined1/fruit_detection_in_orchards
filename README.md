
# Fruit Detection in Orchards
  

## Introduction

This repository is an extension of Faster-RCNN aimed at providing an example of training fruit detection models using [Deep Fruit Detection in Orchards](https://arxiv.org/abs/1610.03677) and [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

Here below, some examples:

![Example Mango 1](https://github.com/zined1/fruit_detection_in_orchards/raw/master/res/20151124T044856.553456_i1990j1463.jpg)

![Example Mango 2](https://github.com/zined1/fruit_detection_in_orchards/raw/master/res/20151124T024807.999518_i1900j868.jpg)

![Example Mango 3](https://github.com/zined1/fruit_detection_in_orchards/raw/master/res/20151124T030416.224953_i1454j433.jpg)

![Example Apple 1](https://github.com/zined1/fruit_detection_in_orchards/raw/master/res/20130320T004803.045018.Cam6_32.jpg)

![Example Apple 2](https://github.com/zined1/fruit_detection_in_orchards/raw/master/res/20130320T004556.375774.Cam6_31.jpg)

## Installation
### Download repo

+ Clone the repository

```Shell
# Make sure to clone with --recursive
git clone --recursive https://github.com/zined1/deep_fruit_detection_in_orchards
```
### Compile

+ Edit your arch in setup.py script to match your GPU

```Shell
cd tf-faster-rcnn/lib
# Change the GPU architecture (-arch) if necessary
# Edit setup.py
```
+ Build the Cython modules
```Shell
make clean
make
```
## Run demo

  + Download and move the [pre-trained models](https://drive.google.com/open?id=1AvTDnrz1GKgo4MejeXWvpyx2kx45R4yR)
```Shell
tar xvf output.tar.gz
mkdir -p ./tf-faster-rcnn/output/vgg16/fruits_dataset_train
mv vgg16_faster_rcnn_iter_40000.* ./tf-faster-rcnn/output/vgg16/fruits_dataset_train/
```
  + Download the dataset
```Shell
wget http://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/acfr-multifruit-2016.zip
unzip acfr-multifruit-2016.zip
```
+ Move the dataset to tf-faster-rcnn/data/fruits_dataset and replace old files (tf-faster-rcnn) with those on this repository

+ Run demo
```Shell
CUDA_VISIBLE_DEVICES=0 python3.6 tools/demo.py
```
Results are written in output.json. An exemple of the output can be

```json
{  
   "20151124T044856.553456_i1990j1463":{  
      "num_objects":6,
      "loc":[  
         [341,339,401,401],
         [158,302,201,356],
         [156,164,189,206],
         [156,92,210,153],  
         [353,21,388,59],  
         [0, 156, 19, 213]
      ]
   },
   "20151124T024807.999518_i1900j868":{  
      "num_objects":3,
      "loc":[  
         [50,470,82,498],
         [328,22,350,44],
         [140,427,166,456]
      ]
   }
}
```

## Further

### Training

If you want to train the model
+ Download pre-train model

 ```Shell
mkdir -p ./tf-faster-rcnn/data/imagenet_weights
cd ./tf-faster-rcnn:data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../
```
  
 + Replace old files (tf-faster-rcnn) with those on this repository
 
+ Run by changing [GPU_ID]
 
```Shell
./tf-faster-rcnn/experiments/scripts/train_faster_rcnn.sh [GPU_ID] fruits vgg16
```
