# DELTA: DEGRADATION-FREE FULLY TEST-TIME ADAPTATION

Fully test-time adaptation aims at adapting a pre-trained model to the test stream during real-time inference, which is urgently required when the test distribution differs from the training distribution. Several efforts have been devoted to improving adaptation performance. However, we find that two unfavorable defects are concealed in the prevalent adaptation methodologies like test-time batch normalization (BN) and self-learning. First, we reveal that the normalization statistics in test-time BN are completely affected by the currently received test samples, resulting in inaccurate estimates. Second, we show that during test-time adaptation, the parameter update is biased towards some dominant classes. In addition to the extensively studied test stream with independent and class-balanced samples, we further observe that the defects can be exacerbated in more complicated test environments, such as (time) dependent or class-imbalanced data. We observe that previous approaches work well in certain scenarios while show performance degradation in others due to their faults. In this paper, we provide a plug-in solution called DELTA for Degradation-freE fuLly Test-time Adaptation, which consists of two components: (i) Test-time Batch Renormalization (TBR), introduced to improve the estimated normalization statistics. (ii) Dynamic Online re-weighTing (DOT), designed to address the class bias within optimization. We investigate various test-time adaptation methods on three commonly used datasets with four scenarios, and a newly introduced real-world dataset. DELTA can help them deal with all scenarios simultaneously, leading to SOTA performance.


## Installation
```
pip install -r requirements.txt
```

## Datasets
ImageNet-C is used in this repository, which can be downloaded from https://zenodo.org/record/2235448#.YpCSLxNBxAc. Then, organize the directories as following:

```
/dockerdata
├── imagenet-c
│   ├── gaussian_noise
│   ├── ...
```


## Fully Test-Time Adaptation

The config files are provided in `./configs/`. And one can perform different test-time adaptation methods with these config files. Some examples are listed below.

- Source, IS+CB scenario
```
python3 main.py --cfg './configs/imagenet/imagenetc_source.yaml' --seed 2020 --distri_type 'iid'
```
  
- Tent, DS+CB scenario
```
python3 main.py --cfg './configs/imagenet/imagenetc_tent.yaml' --seed 2020 --distri_type 'noniid_1.0'
```

- Tent + DELTA, DS+CI scenario
```
python3 main.py --cfg './configs/imagenet/imagenetc_tentdelta.yaml' --seed 2020 --distri_type 'noniidlt_0.5_0.1'
```


## Citation

If you find DELTA is helpful in your research, please consider citing our paper:
```
@inproceedings{
zhao2023delta,
title={{DELTA}: {DEGRADATION}-{FREE} {FULLY} {TEST}-{TIME} {ADAPTATION}},
author={Bowen Zhao and Chen Chen and Shu-Tao Xia},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=eGm22rqG93}
}
```



## Acknowledgment
The code is modified from [EATA](https://github.com/mr-eggplant/EATA), [Tent](https://github.com/DequanWang/tent), and [TTT](https://github.com/yueatsprograms/ttt_imagenet_release).

