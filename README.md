# TRSADepth: Self-Supervised Monocular Depth Estimation with Texture-Robust Sparse Attention in Natural Environments

## Train
```shell
python train.py
--model_name [TRSADepth]
--split [eigen_zhou]
--dataset [kitti]
--pose_model_type [resnet]
--data_path [E:\dataset\KITTI]
--frame_ids 0 -1 1
--batch_size [8/16]
--png
--height [384]
--width [640]
--num_epochs [15/20/...]
--log_dir [./logs/kitti]
```
## Evaluation
```shell
python [eval.py]
--eval_split [eigen] 
--dataset [kitti]
--data_path [data_path]
--load_weights_folder [...]
--eval_mono
```

## 可视化

```shell
tensorboard --logdir [./logs/kitti]
tensorboard --logdir ./logs/kitti
```


## DataSet

[UAV_ula](https://github.com/takisu0916/UAV_ula)

[UAVid](https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-zux-xqv4)

[KITTI](https://www.cvlibs.net/datasets/kitti/)

[Make3D](http://make3d.cs.cornell.edu/data.html#make3d)

[Our Dataset is available at  BaiduYun:0000](https://pan.baidu.com/s/1-nvNVd_udJl2fwIwfbpODg?pwd=0000)

## Result
|    Dataset     | abs_rel | sq_rel | rmse  | $\text{rmse}_{log}$ |  a1   |  a2   |  a3   |                                              Weights                                               | 
|:--------------:|:-------:|:------:|:-----:|:-------------------:|:-----:|:-----:|:-----:|:--------------------------------------------------------------------------------------------------:| 
|   KITTI(640)   |  0.103  | 0.662  | 4.338 |        0.176        | 0.895 | 0.968 | 0.985 | [Weights](https://drive.google.com/drive/folders/145gizdY9htn3XET5Ku5yyQOd5IRyn_Aq?usp=drive_link) | 
|  KITTI(1024)   |  0.097  | 0.628  | 4.189 |        0.171        | 0.904 | 0.970 | 0.985 |  [Weights](https://drive.google.com/drive/folders/1cUGbttC-wHtDRDq8x6Y_PZAyDzSLcYS-?usp=sharing)   | 
|   UAVid(All)   |  0.063  | 0.100  | 0.667 |        0.102        | 0.968 | 0.991 | 0.995 |  [Weights](https://drive.google.com/drive/folders/1bxPcY1USAs1Mfa8kfVWkOt5sr2-sgavx?usp=sharing)   | 
|  UAVid(China)  |  0.094  | 0.311  | 1.582 |        0.153        | 0.927 | 0.980 | 0.990 |  [Weights](https://drive.google.com/drive/folders/1t4CdJwSRRyPRHO0WGbxK4GSRvtPTu-uC?usp=sharing)   | 
| UAVid(Germany) |  0.053  | 0.044  | 0.378 |        0.082        | 0.981 | 0.995 | 0.997 |  [Weights](https://drive.google.com/drive/folders/1PPPVEe4gDavlqeHMmTQBcMftd-0WXM0W?usp=sharing)   |
|    UAV_ula     |  0.080  | 0.082  | 0.476 |        0.105        | 0.922 | 0.986 | 0.996 |  [Weights](https://drive.google.com/drive/folders/1OvDBQ_msIewH0943KSahhWuPn5J9n1g2?usp=sharing)   |

## Acknowledgement
Thanks the authors for their works:

[MonoDepth2](https://github.com/nianticlabs/monodepth2 )

[EfficientViT](https://github.com/mit-han-lab/efficientvit )

[MonoViT](https://github.com/zxcqlf/MonoViT )

[HR-Depth]( https://github.com/shawLyu/HR-Depth )