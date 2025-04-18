# defocused_segmentation

We offer a solution of soft semantic segmentation for video and photo background matting.

The dataset we used is [Background Matting V2](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).


## 1. Dataset
The dataset provides 484 pairs of high-resolution alpha matte and foreground video clips, constituting 240,709 unique frames. The alpha matte and foreground data were extracted from green screen stock footage. Due to licensing, the dataset only provides foreground layers(human foreground) and associated alpha masks. Hence, it is worth investigating if we want to composite human foreground with provided background images (around 200) when synthesizing our train set.



## 2. Synthesize Data
We tested using videos with only black backgrounds to synthesize a shallow depth of field blur effect, however, due to the lack of background, the depth estimation, which is essential for our simulation, can be flickering and unstable. Therefore, we first randomly composite videos with provided background images.  

To simulate a DoF blur, we used depth-aware Gaussian blur so that the whole blur is calculate similar to how photographical DoF is. For stable and temporal consistency, we estimate the subject depth in a video by Video-Depth-Anything, one of the latest depth estimation work specifically oriented towards sequence data.
