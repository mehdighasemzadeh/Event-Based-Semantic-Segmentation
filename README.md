# Event-Based Semantic Segmentation

This work proposes a pyramid-based network for semantic segmentation using an event camera, we evaluate our model on DDD17 dataset[[link](https://github.com/Shathe/Ev-SegNet#readme)] and Event-Scape dataset[[link](https://rpg.ifi.uzh.ch/RAMNet.html)] 

# Network Architecture

<img src="pic/eventmod.png" width="550" height="350">

# Requirements
* Python 3.7+
* Tensorflow 2.8 + 
* Opencv
* Keras
* segmentation-models


# Dataset

## DDD17 dataset

**The semantic segmentation classes in DDD17 dataset:** flat, construction+sky, object,  nature,  human, and vehicle

**A video from results:**

[![Watch the video](pic/rec1487417411_export_3772.png)](https://youtu.be/AL911t6QpBA)

P: Our network prediction, GT: Ground Truth

**Training**

Please download DDD17 dataset from [here](https://github.com/Shathe/Ev-SegNet) or [here](https://drive.google.com/file/d/1XEUfhho-2g8NH3AYT49zBhDjybHOWAkF/view?usp=sharing), then extract it in DDD17/Dataset directory
```
cd DDD17
python3 train.py
```

**Pre-trained Weights**

Please download Weights from [here](https://drive.google.com/file/d/15K_s0RYAuEi4DkH-mfuFvMq7Qp6yQwcX/view?usp=sharing) , then extract it in DDD17 directory

**Evaluating**

For revealing the network performance, eval.py creates and saves results in DDD17/output directory
```
cd DDD17
python3 eval.py
```




## Event-Scape dataset

**The semantic segmentation classes in Event-Scape dataset:** ‫‪Unlabeled‬‬‫‪ +‬‬ ‫‪Sky,‬‬ ‫‪Wall‬‬‫‪ +‬‬ ‫‪Fence‬‬ ‫‪+‬‬ ‫‪Building‬‬, Person‬‬, sign‬‬‫‪Traffic‬‬ ‫‪+‬‬ ‫‪Pole‬‬, ‫‪Road‬‬, ‫‪Sidewalk‬‬, Vegetation‬‬, Vehicle‬‬,

**A video from results:**

[![Watch the video](pic/05_001_0162_image.png)](https://youtu.be/Q1pNcZDNzos)

P: Our network prediction, GT: Ground Truth

**Training**

The original Event-Scape dataset is available [here](https://github.com/Shathe/Ev-SegNet) for training and evaluating the network, download the customized Event-Scape dataset from [here](https://drive.google.com/file/d/1XEUfhho-2g8NH3AYT49zBhDjybHOWAkF/view?usp=sharing), then extract it in Event-Scape/Dataset directory
```
cd Event-Scape
python3 train.py
```

**Pre-trained Weights**

Please download Weights from [here](https://drive.google.com/file/d/1OHDY8iooyAwIlNHPBKU-VPCIBNuFedhK/view?usp=sharing) , then extract it in Event-Scape directory

**Evaluating**

For revealing the network performance, eval.py creates and saves results in Event-Scape/output directory
```
cd Event-Scape
python3 eval.py
```






