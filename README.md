# EfficientSAM
EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

https://github.com/yformer/EfficientSAM



## Online Demo & Examples
Online demo and examples can be found in the [project page](https://yformer.github.io/efficient-sam/).


## Model
EfficientSAM checkpoints are available under the weights folder of this github repository. Example instantiations and run of the models can be found in [EfficientSAM_example.py](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py).

| EfficientSAM-S | EfficientSAM-Ti |
|------------------------------|------------------------------|
| [Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) |[Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt)|

You can directly use EfficientSAM with checkpoints,
```
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
efficientsam = build_efficient_sam_vitt()
```

## Jupyter Notebook Example
The notebook is shared [here](https://github.com/yformer/EfficientSAM/blob/main/notebooks)


## Acknowledgement

+ [SAM](https://github.com/facebookresearch/segment-anything)
+ [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
+ [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
+ [U-2-Net](https://github.com/xuebinqin/U-2-Net)


If you're using EfficientSAM in your research or applications, please cite using this BibTeX:
```bibtex


@article{xiong2023efficientsam,
  title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
  author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
  journal={arXiv:2312.00863},
  year={2023}
}
```
