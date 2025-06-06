# defocused_segmentation

This is a researching-driven project, testing the potential of using Discrete Wavelet Transform for high-fidelity alpha matting via defocused cues. 
Due to the limitation of computation resources, we are heavily constrained on the iteration speed, which limits our experiment amount on hyperparameter tuning, comparison study and ablation study. However, our major focus for this project is to test if DWT can be tailored to such a task, given the limited experiments have already shown potentials, we are happy with the overall progress. Later on, we will focus on designing better training strategies as well as comprehensive experiments. Since we are the first to use DWT in matting task, so the whole development process involved in testing at least 5 major versions of model framework as well as training strategies (and loads of sub versions). In the repo, we only put some of the latest models and pipeline scripts to avoid redundancy. More attempts we have tried can be found in other branches.

The dataset we used is derived from [Background Matting 240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).


## 1. Environment Configuration
Due to package conflicting, we adopted two environment configurations for our project.

To create our defocused dataset:
Clone Video-Depth-Anything repo from its official github page;
Follow the instruction to install the environment, then activate the environment.

To run our model:
```
chmod +x env_setup.sh
./env_setup.sh
conda activate sam2_matanyone
```

Our environments in addition directly fetch dependencies such as SAM2, MatAnyone, VideoDepthAnything by downloading their git repo. Please replace the `MatAnyone/matanyone/utils/get_default_model.py` by the python file with the same name under the project root.


## 2. Synthesize Data
The VideoMatte240K dataset provides 484 pairs of high-resolution alpha matte and foreground video clips, constituting 240,709 unique frames. The alpha matte and foreground data were extracted from green screen stock footage. Due to licensing, the dataset only provides foreground layers(human foreground) and associated alpha masks. Hence, it is worth investigating if we want to composite human foreground with provided background images (around 200) when synthesizing our train set.  
We tested using videos with only black backgrounds to synthesize a shallow depth of field blur effect, however, due to the lack of background, the depth estimation, which is essential for our simulation, can be flickering and unstable. Therefore, we first randomly composite videos with provided background images.  
To simulate a DoF blur, we used depth-aware Gaussian blur so that the whole blur is calculate similar to how photographical DoF is. For stable and temporal consistency, we estimate the subject depth in a video by Video-Depth-Anything, one of the latest depth estimation work specifically oriented towards sequence data.

To compose human rgb foregrounds with random background images, 
```
conda activate defocused-env
cd preprocess
python compose_video_bg.py
```

To apply Depth-of-Field blur synthesis for our final synthetic dataset, please run the provided shell script. This shell script is built by dynamically estimating calculation chunks depending on the GPU and CPU memory to avoid OOM error. We additionally build a kill-resume mechanism in the shell script to avoid bad garbage management or unexpected crashes.
```
conda activate defocused-env
cd <to project root>
chmod +x run_blur.sh
./run_blur.sh
```

## 3. Framewise Data Arrangement and Coarse Guide Mask Generation
To extract the human foreground coarse mask as a semantic guide for our framework, we use YOLOV8+SAM2. Given 'person' token to 'YOLOV8', it will return the bounding boxes for all human subjects in a given frame. We then use these bounding boxes as input to SAM2. SAM2 will return separate masks for each subject, we then combine all the masks to a single one as our guide mask.

First, we need to arrange our dataset from original video format to per-frame .png.
The dataset is arranged as:
```
<data_folder>
     | -- test
            | -- alpha
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
            | -- fgr
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
     | -- train
            | -- alpha
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
            | -- fgr
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4

```
To convert such data arrangement,
```
conda activate sam2_matanyone
chmod +x run_frame_extraction.sh
RATIO=0.5 ./run_frame_extraction.sh  
```
where 0.5 is the ratio of frames/total_frames for each video. This is to help us randomly sample data subsets. 0.5 means 50%.

Then we could apply coarse mask extraction:
```
conda activate sam2_matanyone
chmod +x run_mask_extraction.sh
./run_mask_extraction.sh first_frame or ./run_mask_extraction.sh all_frames
```
The argument basically allows us to extract the guide mask on all the video frames or just the first frame for each video.


## 3. Training/Evaluation
Under the `model` directory, `refiner_dwt_maggie.py`, `refiner_dwt_maggie_heavydwt.py` are the ones we used to run experiments.  
* `refiner_dwt_maggie.py` adopts a lightweight differentiable DWT
* `refiner_dwt_maggie_heavydwt.py` utilizes a heavier DWT and supports different kernel for DWT (db1, db2, db4).
Other models are intermidiate versions as we progressively developed to our final version of models.

Under the `pipeline` directory, there are two files. The one with `v0001` is a full version; however, the strategy within suppress model behavior on obtaining high frequency details.
The other one is still working on progress to adjust the strategy for better low-freqeuency structure correction and high-frequency detail recovery.

To run training,
```
conda activate sam2_matanyone
cd pipeline
python solver_refiner_maggie.py

```

To run validation,
`cd` to the project root directory;
```
python inference.py --i <image_path> --c <checkpoint_path> --o <image_output_path> --resize_h(opt) 720 --resize_w(opt) 1080
```
