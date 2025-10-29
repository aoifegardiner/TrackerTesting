# WAFT

[[Paper](https://arxiv.org/abs/2506.21526)]

We introduce Warping-Alone Field Transforms (WAFT), a simple and effective
method for optical flow. WAFT is similar to RAFT but replaces cost volume
with high-resolution warping, achieving better accuracy with lower memory cost.
This design challenges the conventional wisdom that constructing cost volumes is
necessary for strong performance. WAFT is a simple and flexible meta-architecture
with minimal inductive biases and reliance on custom designs. Compared with
existing methods, WAFT ranks 1st on Spring and KITTI benchmarks, achieves
the best zero-shot generalization on KITTI, while being up to 4.1x faster than
methods with similar performance.

<img src="assets/Vis.png" width='1000'>

If you find WAFT useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2506.21526">
        WAFT: Warping-Alone Field Transforms for Optical Flow
    </a>
</h3>
<p align="center">
    <a href="https://memoryslices.github.io/">Yihan Wang</a>,
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@misc{wang2025waftwarpingalonefieldtransforms,
      title={WAFT: Warping-Alone Field Transforms for Optical Flow}, 
      author={Yihan Wang and Jia Deng},
      year={2025},
      eprint={2506.21526},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.21526}, 
}
```

## Requirements
Our code is developed with pytorch 2.7.0, CUDA 12.8 and python 3.12. 
```Shell
conda create --name waft python=3.12
conda activate waft
pip install -r requirements.txt
```

Please also install [xformers](https://github.com/facebookresearch/xformers) following instructions.

## Model Zoo

Google Drive: [link](https://drive.google.com/drive/folders/1qimz12pIEwktiBYwYtPQcCrYykBHw-bX?usp=sharing).

## Datasets
To evaluate/train WAFT, you will need to download the required datasets: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/). Please also check [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) for more details.

## Training

Please prepare the [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) checkpoints in the `depth-anything-ckpts` folder before you start training.

```Shell
python train.py --cfg config/chairs.json
python train.py --cfg config/chairs-things.json --restore_ckpt ckpts/chairs.pth
```

## Evaluation & Submission

```Shell
python evaluate --cfg config/chairs-things.json --ckpt ckpts/chairs-things.pth --dataset sintel
python submission --cfg config/tar-c-t-kitti.json --ckpt ckpts/tar-c-t-kitti.pth --dataset kitti
```

## Acknowledgements

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [DPT](https://github.com/isl-org/DPT), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), and [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2). We thank the original authors for their excellent work.
