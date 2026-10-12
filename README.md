# Human Pose Estimation Literature

A curated list of recent literature on human pose estimation with computer vision methods.

## Contents

- [Capability Comparison](#capability-comparison)
- [Multi-person 2D Keypoint Estimation](#multi-person-2d-keypoint-estimation)
  - [AlphaPose](#alphapose-whole-body-regional-multi-person-pose-estimation-and-tracking-in-real-time)
- [Monocular 3D Human Mesh Recovery](#monocular-3d-human-mesh-recovery)
  - [SAM 3D Body](#sam-3d-body-robust-full-body-human-mesh-recovery)
  - [Fast SAM 3D Body](#fast-sam-3d-body-accelerating-sam-3d-body-for-real-time-full-body-human-mesh-recovery)
- [Marker-based Motion Capture](#marker-based-motion-capture)
  - [DirtyMoCap](#dirtymocap-robust-motion-capture-from-unconstrained-markers)

## Capability Comparison

| | AlphaPose | SAM 3D Body | Fast SAM 3D Body | DirtyMoCap |
|---|---|---|---|---|
| **Venue** | TPAMI 2022 | CVPR 2026 (oral) | ECCV 2026 | SIGGRAPH Asia 2026 |
| **Input** | RGB image / video | Single RGB image (+ optional 2D keypoint / mask prompts) | RGB image / video stream; single- or multi-view | Optical marker point clouds (sparse, noisy, unordered, any marker layout) |
| **People** | Multi-person (detector + top-down) | Multi-person (detector + per-person crops) | Multi-person (parallel multi-crop) | Single subject per marker set |
| **2D landmarks** | ✅ Main output: COCO 17, Halpe 26, or Halpe 136 whole-body | ✅ 70 keypoints (MHR70), projected from 3D | ✅ Inherited from 3DB (MHR70) | ❌ (no image input) |
| **3D landmarks** | ⚠️ Only via the optional HybrIK/SMPL model (24 SMPL joints) | ✅ 70 keypoints + full MHR skeleton joints | ✅ MHR70 + SMPL joints (24) from the MHR→SMPL mapping | ✅ SMPL-H joints, fitted to proxy anchors |
| **3D mesh / body model** | ⚠️ SMPL via HybrIK (v0.6+) | ✅ MHR (Momentum Human Rig) | ✅ MHR + feedforward MHR→SMPL | ✅ SMPL-H |
| **Body (torso + limbs)** | ✅ | ✅ | ✅ | ✅ |
| **Feet** | ✅ 6 foot keypoints (Halpe 26/136) | ✅ 6 foot keypoints (big toe, small toe, heel) | ✅ (MHR); SMPL output has only coarse foot joints | ⚠️ SMPL-H ankle/foot joints only |
| **Hands / fingers** | ✅ 21 keypoints per hand (Halpe 136) | ✅ 21 keypoints per hand + articulated MHR hands; dedicated hand decoder | ✅ MHR hands; ❌ in SMPL output (SMPL has no fingers) | ✅ SMPL-H articulated fingers (if markers allow) |
| **Face** | ✅ 68 face landmarks (Halpe 136) | ⚠️ 5 keypoints (nose, eyes, ears); MHR has 72 expression params, but the released head sets them to zero | ⚠️ Same as 3DB | ❌ No face/expression in SMPL-H |
| **Body shape** | ⚠️ SMPL β (HybrIK only) | ✅ 45 MHR shape params + scale | ✅ MHR shape; SMPL β via mapping | ✅ SMPL-H β |
| **Global camera / translation** | ❌ (2D) / ⚠️ camera-relative (HybrIK) | ✅ Camera translation + focal length (FoV from MoGe2) | ✅ Same as 3DB | ✅ World translation (mocap frame) |
| **Temporal tracking** | ✅ Pose Flow / Pose-Aware Identity Embedding | ❌ Per image | ⚠️ Per frame; optional temporal smoother MLP in MHR→SMPL | ✅ Sliding-window anchor tracking over long sequences |
| **Speed** | Real time | Several seconds per image | ~65 ms per frame end to end (RTX 5090); up to 10.9× faster than 3DB | CUDA Gauss-Newton solver up to 100× faster than PyTorch |
| **Code / weights** | ✅ [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) | ✅ [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body): code, checkpoints and dataset (SAM License) | ✅ [yangtiming/Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) | ⏳ [WangLongZJU/DirtyMoCap](https://github.com/WangLongZJU/DirtyMoCap): announced, not yet released |

Legend: ✅ supported · ⚠️ partial / optional / with caveats · ❌ not supported · ⏳ pending

Notes:
- **MHR70 keypoints** (SAM 3D Body / Fast SAM 3D Body): 5 face (nose, eyes, ears), 12 body (shoulders, elbows, hips, knees, ankles), 6 feet, 2 × 21 hand (incl. wrists), plus olecranon, cubital fossa, acromion (each side) and neck. Taken from `sam_3d_body/metadata/mhr70.py` in the official repo.
- **Halpe 136** (AlphaPose): 26 body (COCO 17 + head, neck, hip, 6 foot points) + 68 face + 42 hand keypoints.
- **SMPL vs. SMPL-H vs. MHR**: SMPL has 24 joints and no fingers or face. SMPL-H adds articulated hands. MHR covers body, hands and face expression, with skeleton and surface shape decoupled.

## Multi-person 2D Keypoint Estimation

### AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time

- **Authors:** Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu, Yong-Lu Li, Cewu Lu (Shanghai Jiao Tong University)
- **Venue:** IEEE TPAMI 2022 (arXiv Nov 2022); builds on RMPE (ICCV 2017)
- **Links:** [arXiv:2211.03375](https://arxiv.org/abs/2211.03375) · [Code](https://github.com/MVIG-SJTU/AlphaPose) · [Project](http://www.mvig.org/research/alphapose.html)
- **Input → Output:** RGB images / video → per-person 2D whole-body keypoints (face, body, hands, feet) with identities tracked across frames; optional 3D SMPL via HybrIK
- **Summary:** A top-down, real-time multi-person system for joint whole-body pose estimation and tracking. It introduces Symmetric Integral Keypoint Regression (SIKR) for fast, precise localization, Parametric Pose NMS to remove redundant detections, and a Pose-Aware Identity Embedding for joint pose estimation and tracking. Training adds a Part-Guided Proposal Generator and multi-domain knowledge distillation. The authors also release the Halpe-FullBody dataset (136 keypoints). Evaluated on COCO-WholeBody, COCO, PoseTrack and Halpe-FullBody.
- **Tags:** `2D` `multi-person` `whole-body` `tracking` `real-time` `dataset`

## Monocular 3D Human Mesh Recovery

### SAM 3D Body: Robust Full-Body Human Mesh Recovery

- **Authors:** Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt Feiszli, Jitendra Malik, Piotr Dollár, Kris Kitani (Meta Superintelligence Labs)
- **Venue:** CVPR 2026 (oral); released Nov 2025, arXiv Feb 2026
- **Links:** [arXiv:2602.15989](https://arxiv.org/abs/2602.15989) · [Code](https://github.com/facebookresearch/sam-3d-body) · [Meta AI](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/) · [Dataset](https://huggingface.co/datasets/facebook/sam-3d-body-dataset) · [MHR](https://github.com/facebookresearch/MHR)
- **Input → Output:** single RGB image (optionally prompted with 2D keypoints or masks) → per-person MHR mesh, 70 2D/3D keypoints, body/hand pose, shape, camera translation and focal length
- **Summary:** A promptable encoder-decoder model (DINOv3-H+ or ViT-H backbone) for single-image full-body human mesh recovery, covering body, feet and hands. A separate hand decoder can refine the hands. It uses the Momentum Human Rig (MHR), a new parametric model that decouples skeletal structure from surface shape. Training data comes from a multi-stage annotation pipeline: manual keypoints, differentiable optimization, multi-view geometry, dense keypoint detection, and a data engine that mines rare poses and viewpoints. Reports state-of-the-art results on 3DPW, EMDB, RICH, COCO, LSPET and FreiHAND.
- **Tags:** `3D` `HMR` `single-image` `promptable` `whole-body` `hands` `MHR` `dataset`

### Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery

- **Authors:** Timing Yang, Sicheng He, Hongyi Jing, Jiawei Yang, Zhijian Liu, Chuhang Zou, Yue Wang (USC Physical Superintelligence Lab, UC San Diego, NVIDIA, Meta Reality Labs)
- **Venue:** ECCV 2026 (arXiv Mar 2026)
- **Links:** [arXiv:2603.15603](https://arxiv.org/abs/2603.15603) · [Project Page](https://yangtiming.github.io/Fast-SAM-3D-Body-Page/) · [Code](https://github.com/yangtiming/Fast-SAM-3D-Body)
- **Input → Output:** RGB image / video stream (single or multi-view) → SAM 3D Body outputs (MHR mesh, keypoints) plus SMPL parameters
- **Summary:** A training-free acceleration of SAM 3D Body. It decouples serial spatial dependencies and prunes the architecture, so multi-crop feature extraction runs in parallel and transformer decoding is streamlined. A YOLO-Pose 2D prior, TensorRT and `torch.compile` speed things up further. The iterative MHR→SMPL fitting is replaced by a feedforward network (more than 10,000× faster), with optional multi-view fusion and temporal smoothing. The paper reports up to a 10.9× end-to-end speedup with on-par accuracy, and better than 3DB on LSPET. Demonstrated in real-time, vision-only teleoperation of a Unitree G1 humanoid (~65 ms per frame on an RTX 5090) and for collecting manipulation-policy data.
- **Tags:** `3D` `HMR` `real-time` `SMPL` `MHR` `robotics` `teleoperation`

## Marker-based Motion Capture

### DirtyMoCap: Robust Motion Capture from Unconstrained Markers

- **Authors:** Long Wang, Shuting Zhao, Shen Yan, Siyuan Yu, Xiaoben Li, Zeyu Cai, Yumeng Hou, Yuliang Xiu
- **Venue:** SIGGRAPH Asia 2026 (arXiv Sep 2026)
- **Links:** [Project Page](https://wanglongzju.github.io/DirtyMoCap-Project-Page/) · [arXiv:2609.19927](https://arxiv.org/abs/2609.19927) · [Code (announced)](https://github.com/WangLongZJU/DirtyMoCap)
- **Input → Output:** sparse, noisy, unordered optical marker point clouds (arbitrary or unknown marker layouts) → 4D SMPL-H motion (pose, translation, shape)
- **Summary:** Unordered marker observations are mapped onto a fixed set of proxy anchors (skeletal joints and body surface points). A sliding-window tracker follows these anchors over long sequences, and a differentiable, learnable Gauss-Newton solver fits SMPL-H to them. A single trained model generalizes across marker configurations and outperforms configuration-specific baselines in joint and vertex reconstruction accuracy. A custom CUDA solver is reported to be up to 100× faster than a standard PyTorch implementation. Applied to raw optical recordings of Chinese martial arts, the method yields a Kung Fu motion dataset of temporally coherent SMPL-H reconstructions.
- **Tags:** `motion-capture` `optical-markers` `SMPL-H` `body-model-fitting` `4D` `dataset`
