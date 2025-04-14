# defocused_segmentation

We offer a solution of soft semantic segmentation for video and photo background matting.

The dataset we used is [Background Matting V2](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).


## 1. Dataset
The dataset provides 484 pairs of high-resolution alpha matte and foreground video clips, constituting 240,709 unique frames. The alpha matte and foreground data were extracted from green screen stock footage. Due to licensing, the dataset only provides foreground layers(human foreground) and associated alpha masks. Hence, it is worth investigating if we want to composite human foreground with provided background images (arounf 200) when synthesizing our train set.



## 2. Synthesize Data
