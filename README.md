# Human Pose Estimation Literature

A curated list of recent literature on human pose estimation with computer vision methods.

## Contents

- [Marker-based Motion Capture](#marker-based-motion-capture)

## Marker-based Motion Capture

### DirtyMoCap: Robust Motion Capture from Unconstrained Markers

- **Authors:** Long Wang, Shuting Zhao, Shen Yan, Siyuan Yu, Xiaoben Li, Zeyu Cai, Yumeng Hou, Yuliang Xiu
- **Year:** 2026 (arXiv, Sep 2026)
- **Links:** [Project Page](https://wanglongzju.github.io/DirtyMoCap-Project-Page/) · [arXiv:2609.19927](https://arxiv.org/abs/2609.19927) · [Code](https://github.com/WangLongZJU/DirtyMoCap)
- **Input → Output:** sparse, noisy, unordered optical marker point clouds (arbitrary/unknown marker layouts) → SMPL-H body pose, translation and shape
- **Summary:** Unordered marker observations are mapped onto a fixed set of proxy anchors (skeletal joints and body surface points). A sliding-window tracker follows these anchors over long sequences, and a differentiable Gauss-Newton solver fits SMPL-H to them. A single trained model generalizes across marker configurations and outperforms configuration-specific baselines in joint and vertex reconstruction accuracy. A custom CUDA solver is reported to be up to 100× faster than a standard PyTorch implementation. Applied to raw optical recordings of Chinese martial arts, the method yields a Kung Fu motion dataset of temporally coherent SMPL-H reconstructions.
- **Tags:** `motion-capture` `optical-markers` `SMPL-H` `body-model-fitting` `dataset`
