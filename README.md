# FG-Mamba

Official research code package for **FG-Mamba: A Boundary-Guided and
Direction-Adaptive State Space Model for Hyperspectral Image Classification**.

FG-Mamba is a center-pixel hyperspectral image classifier with three main
components:

1. a depth-dependent Gaussian center prior and PCA/Sobel boundary cue;
2. four shared-parameter directional Mamba scans with spatially adaptive fusion;
3. cross-layer center-token aggregation with Gaussian-weighted local context.

## Environment

The reported experiments used:

- PyTorch 2.2.2
- CUDA 11.8
- NVIDIA GeForce RTX 4070
- 25 x 25 input patches
- embedding dimension 128
- six FG-Mamba blocks
- 300 epochs, batch size 64, AdamW with learning rate 0.0005 and weight decay 0.0001
- 10-epoch linear warm-up followed by cosine decay
- inverse-frequency class-weighted focal loss (gamma 2) plus projection orthogonality (0.001)

Example installation:

```bash
conda create -n fg-mamba python=3.10 -y
conda activate fg-mamba
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset preparation

Download WHU-Hi-HongHu, WHU-Hi-HanChuan, and Indian Pines from their official
providers.
https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
