# Reproduce The Results
We reproduce the results from this paper. We run 2 experiments:
1. RGB pipeline analysis for different inputs
2. Best model - final training

# Our reproduc Results
<div align="center">

| Model                     | Train Accuracy | Validation Accuracy | Validation Accuracy (Reproduce) |
|---------------------------|----------------|----------------------|----------------------------------|
| RGB frames               | 75.06%         | 78.5%               | 78.12%                          |
| Skeletons with background | 73.5%          | 80%                 | 84.9%                           |
| Skeletons without background | 87.88%      | 84.75%              | 85.68%                          |

**Table 1: Results for RGB pipeline analysis for different inputs**

|         Results         |  Train Accuracy  |  Test Accuracy  |
|:-----------------------:|:----------------:|:---------------:|
|     Author results      |      92.37%      |      90.25%     |
|  Our reproduce results  |      92.68%      |      87.75%     |

**Table 2: RComparison of Author and Reproduced Results**

</div>

# New Updates to the Implementation for Reproduction
- Apply multithread to load data into `dataloader` (see [this notebook](Violence-Detection-With-Human-Skeletons/experiments/RWF-2000/notebooks/1.%20RGB%20pipeline%20analysis/1.%20Results%20for%20different%20inputs/Original%20videos.ipynb)
)
- Release our best model for this task (87.75%) ([best model](Violence-Detection-With-Human-Skeletons/experiments/RWF-2000/notebooks/best_model_no_bg_50epoch.keras)).


# Some advises for reproducing the results
1. Install openpose
- We follow exactly the instructions from the blog that the authors suggest ([blog](https://amir-yazdani.github.io/post/openpose/))
2. Install requirements:
- check the tensorflow and keras version
3. Running on GPU
- Check your GPU ID when running the original code. We run on only 1 GPU which `index 0` so we changed a little bit when initializing GPU .


# Human Skeletons and Change Detection for Efficient Violence Detection in Surveillance Videos

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/human-skeletons-and-change-detection-for/activity-recognition-on-rwf-2000)](https://paperswithcode.com/sota/activity-recognition-on-rwf-2000?p=human-skeletons-and-change-detection-for)

This is the GitHub repository associated with the paper [Human Skeletons and Change Detection for Efficient Violence Detection in Surveillance Videos](https://www.sciencedirect.com/science/article/pii/S1077314223001194), published in Computer Vision and Image Understanding (CVIU), vol. 233, 2023. The method achieves 90.25% accuracy in the RWF-2000 validation set with just 60k trainable parameters.

<p align="center">
  <img src="demo/architecture.png">
</p>

The purpose of this repository is to provide the code needed to replicate the results of the paper. Moreover, the training logs of all the experiments are also included.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#preprocessing">Preprocessing</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#access-to-model-weights">Access to model weights</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## Demo

![](demo/Will_Smith.gif)

![](demo/murcia_fight.gif)

To demonstrate the ability of our proposal to detect violence in real-life scenarios, we have executed our best model with two videos that are not present in any of the datasets. In the second video, a girl is brutally beaten outside of a bar in Murcia, a city in southern Spain ([news report](https://www.laopiniondemurcia.es/murcia/2017/01/23/brutal-paliza-chica-centro-murcia-31944907.html)).

## Prerequisites

### Conda environment

The `enviroment.yml` file contains all the dependencies used during the development of this project. You can create the corresponding conda environment by running the following command:

```bash
conda env create -f environment.yml
```

### Datasets

If you would like to train or validate a model, you will need one of the datasets that were used in the paper. Find below their links:

* [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection): as stated in the paper, this is the main dataset that we have used to train and validate our model. The authors of the original dataset require the signing of an [Agreement Sheet](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection#download) to grant access.

* [Hockey](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes)

* [Movies](https://academictorrents.com/download/70e0794e2292fc051a13f05ea6f5b6c16f3d3635.torrent)

* [Crowd](https://www.openu.ac.il/home/hassner/data/violentflows/)

### OpenPose

Apart from the data, one of the essential components of our proposal is the skeletons' detector. As mentioned in the paper, we use OpenPose. Please, follow their [instalation guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) to compile the library and be able to extract skeletons. A complementary installation guide is available at [this link](https://amir-yazdani.github.io/post/openpose/). 

## Preprocessing

Once you have compiled the OpenPose library, you can use our `preprocess.py` script to extract the skeletons from the videos of your choice. The script also applies a gamma correction before extracting the skeletons, which can be disabled.

## Training

All the notebooks required to reproduce the results of the paper in terms of training and validation are included in the [`experiments`](https://github.com/atmguille/Violence-Detection-With-Human-Skeletons/tree/main/experiments) directory. The training logs with the results reported in our paper are provided inside each notebook. For the RWF-2000 dataset, a summary of the results is also provided in the [`experiments/RWF-2000/results_summary`](https://github.com/atmguille/Violence-Detection-With-Human-Skeletons/tree/main/experiments/RWF-2000/results_summary) directory. Note that some extra, less important, experiments not reported in the paper are included in these directories for this last dataset.

## Inference

Once you have trained a model, you can try to detect violence in a video of your choice. First of all, you have to use the `preprocess.py` script to extract the skeletons. Afterward, you can use the `inference.ipynb` notebook to perform the inference. This notebook will output a sequence of frames with the predicted probabilites of violence. Find an example of a frame with a predicted probability below: 

<p align="center">
  <img src="demo/prediction_probability.png">
</p>

With this frames, you can render a video with the aggreagted predictions using [FFmpeg](https://ffmpeg.org/), which can then be merged with the original video to obtain what is shown in the Demo. The command is:
    
```bash
ffmpeg -i predictions/VIDEO_NAME-%d.png -r 10 output.mp4
```

## Access to model weights

The primary goal of this repository is to facilitate the replication of our research and provide tools to enhance and expand upon it. We are confident that the provided resources are sufficient for this purpose. Intentionally, we refrain from publishing the model weights to prevent any unauthorized commercial usage. If you believe that your research could benefit from accessing the model weights or if you wish to discuss potential commercial applications, kindly complete this [Form](https://docs.google.com/forms/d/1mK6DpStHeDJQk7zuTiHQfwAK_g0WyJsTaGNFyGHw0g4).

## Citation

Please cite our paper if this work helps your research:

```
@article{GARCIACOBO2023SkeletonsViolence,
  title = {Human skeletons and change detection for efficient violence detection in surveillance videos},
  journal = {Computer Vision and Image Understanding},
  volume = {233},
  pages = {103739},
  year = {2023},
  issn = {1077-3142},
  doi = {https://doi.org/10.1016/j.cviu.2023.103739},
  url = {https://www.sciencedirect.com/science/article/pii/S1077314223001194},
  author = {Guillermo Garcia-Cobo and Juan C. SanMiguel}
}
```
