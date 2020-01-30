# EDSR-pytorch
Pytorch implementation of Enhanced Deep Residual Network for Single Image Super Resolution -- (https://arxiv.org/abs/1707.02921)

### Dependencies:

1) Python = 3.6
2) torch >= 1.0.0
3) Pillow
4) numpy

### Data:

We have used the DIV2K dataset to train our model which can be found [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/https://data.vision.ee.ethz.ch/cvl/DIV2K/). Download the data and arrange the folder in the following manner.

```data```

- train
  - img
  - label
- validation
  - img
  - label
