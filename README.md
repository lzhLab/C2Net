# C<sup>2</sup>Net

Code for the manuscript:

**“Bridging Unknown to Known: Connectivity-Guided Faithful Liver Vessel Segmentation”**  
Yiqi Wang, Jin Zhang, Ziqi Wang, Zhicai Peng, Xunliang Xu, and Liang Zhao.

C<sup>2</sup>Net is a connectivity-guided liver vessel segmentation framework. The current implementation provides a unified training entry point, `train_seg.py`, and supports calling the proposed C<sup>2</sup>Net model as well as additional baseline segmentation models through a common interface.

---

## 1. Dependencies

The original environment used for this project is:

```bash
python 3.8
pytorch 1.8
```

Recommended basic packages:

pip install numpy scipy scikit-image pillow opencv-python matplotlib
Optional packages for additional baseline models:

pip install timm          # commonly used by TransUNet, SCUNet++, SegFormer
pip install torchinfo     # optional, for printing model summaries
pip install monai         # useful for nnU-Net-like and 3D medical segmentation models
pip install scipy scikit-image   # required for ASSD, HD, and clDice evaluation
Note: Only install the optional dependencies required by the models you plan to run.

## 2. Project Structure
A typical project structure is:

```bash
C2Net/
├── train_seg.py
├── trainer.py
├── criterions/
│   ├── __init__.py
│   └── db_criterion.py
├── datasets/
│   ├── dataset_livs.py
│   └── data/
│       ├── LiVS/
│       │   ├── image/
│       │   └── label/
│       ├── MSD/
│       └── 3DIRCADb/
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── C2Net.py
│   └── ...
└── results/
    ├── logs/
    ├── weights/
    ├── predicts/
    └── plots/
```

The proposed model is implemented in:
```
models/C2Net.py
```
In the current implementation, RNN_Model is used as a compatibility wrapper for the proposed C<sup>2</sup>Net architecture.

## 3. Dataset Organization
The datasets used for training and evaluation include: 3DIRCADb, MSD, and LiVS

The input images and ground-truth labels should be organized as follows:
```
datasets/
└── data/
    ├── LiVS/
    │   ├── image/
    │   │   ├── 01/
    │   │   │   ├── 01_1.png
    │   │   │   ├── 01_2.png
    │   │   │   └── ...
    │   │   ├── 02/
    │   │   └── ...
    │   └── label/
    │       ├── 01/
    │       │   ├── 01_1.png
    │       │   ├── 01_2.png
    │       │   └── ...
    │       ├── 02/
    │       └── ...
    ├── MSD/
    │   └── ...
    └── 3DIRCADb/
        └── ...
```
The image folder and label folder should have matched case IDs and slice names.

### 4. Basic Usage
To train the default model, run:
```
python train_seg.py
```
This is equivalent to:
```
python train_seg.py --arch RNN_Model
```
To explicitly train the proposed model:
```
python train_seg.py --arch RNN_Model --lr 1e-3 --epochs 100 --batch-size 8
```
To print the model architecture and hyperparameters:
```
python train_seg.py --arch RNN_Model --print-model --epochs 1 --batch-size 2
```
To train U-Net:
```
python train_seg.py --arch Unet --lr 1e-3 --epochs 100
```

## 5. Supported and Optional Baseline Models
The training script is designed to support multiple baseline segmentation models through a unified interface.

| No. | Model | Reference | Code Link |
|---:|---|---|---|
| 1 | U-Net | Ronneberger et al., MICCAI 2015 | https://github.com/milesial/Pytorch-UNet |
| 2 | PSPNet | Luo et al., Engineering Letters 2024 | https://github.com/hszhao/PSPNet |
| 3 | TransUNet | Chen et al., arXiv 2021 | https://github.com/Beckschen/TransUNet |
| 4 | SA-UNet | Guo et al., ICPR 2021 | https://github.com/clguo/SA-UNet |
| 5 | R2U-Net | Alom et al., arXiv 2018 | https://github.com/navamikairanda/R2U-Net |
| 6 | RU-Net | Wang et al., CMPB 2022 | https://github.com/siml3/RU-Net |
| 7 | LS-FPN | Gao et al., IEEE TMI 2023 | https://github.com/lzhLab/LiVS |
| 8 | nnU-Net | Isensee et al., Nature Methods 2020 | https://github.com/MIC-DKFZ/nnUNet |
| 9 | 3D U-Net | Huang et al., Computers in Biology and Medicine 2018 | https://github.com/wolny/pytorch-3dunet |
| 10 | SCUNet++ | Chen et al., WACV 2024 | https://github.com/justlfc03/scunet-plusplus |
| 11 | UMamba | Jain et al., BSPC 2026 | https://github.com/DJ-CHB/DiffUMamba-Official |

To make a model callable from train_seg.py, each model should follow the same interface.

## 5.1 Recommended Constructor
```
class YourModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32):
        super().__init__()
        ...
```

## 5.2 Recommended Forward Function
For 2D models:
```
def forward(self, x):
    """
    Args:
        x: Tensor with shape [B, C, H, W]

    Returns:
        logits: Tensor with shape [B, out_channels, H, W]
    """
    return logits
```
For 3D models:
```
def forward(self, x):
    """
    Args:
        x: Tensor with shape [B, C, D, H, W]

    Returns:
        logits: Tensor with shape [B, out_channels, D, H, W]
    """
    return logits
```

## 5.3 Required Output
All models should return logits, not probabilities:
```
return logits
```
The loss function applies sigmoid internally when needed.

### 6. Adding a New Baseline Model
To add a new model, follow these steps.

# Step 1: Add the Model File
Example for PSPNet:
```
models/
└── pspnet/
    ├── __init__.py
    └── pspnet.py
```
# Step 2: Add a Wrapper If Needed
Many official implementations use different argument names, such as num_classes, n_channels, or n_classes. Wrap them into the unified interface:
```
import torch.nn as nn


class PSPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.net = OriginalPSPNet(
            in_channels=in_channels,
            num_classes=out_channels
        )

    def forward(self, x):
        return self.net(x)
If the original model returns multiple outputs, return the main segmentation logits:

def forward(self, x):
    outputs = self.net(x)

    if isinstance(outputs, (tuple, list)):
        return outputs[0]

    return outputs
```
# Step 3: Register the Model in models/__init__.py
```
from .unet import Unet
from .C2Net import RNN_Model
from .pspnet.pspnet import PSPNet
```
# Step 4: run train_seg.py
Run:
```
python train_seg.py --arch PSPNet
```

### 7. Model-Specific Notes
# U-Net

U-Net can usually be adapted with minimal changes. The first convolution accepts:
```
in_channels = expand_size * 2 + 1
```

and the final segmentation head outputs:
```
out_channels = output_size
```
# PSPNet

PSPNet usually contains a pyramid pooling module and a final classifier. The adaptations include:

modify the input stem to accept in_channels;

modify the final classifier to output out_channels;

upsample the output to the original image size;

return final logits only.

# TransUNet

TransUNet is configuration-dependent. The adaptations include:

setting the input image size;

setting patch size and ViT configuration;

modifying the input channel setting if the implementation assumes RGB input;

setting the number of output classes to out_channels;

returning final logits only.

# SA-UNet

SA-UNet can be treated as a U-Net variant with spatial attention. The adaptations include:

changing the input channel number;

changing the final output channel number;

ensuring that the model returns logits only.

# R2U-Net

R2U-Net uses recurrent residual blocks with recurrent depth t=2. Required adaptations include:

matching input and output channels;

exposing t=2 as a fixed or configurable argument;

returning one logits tensor.

# RU-Net

RU-Net implementations may differ across repositories. The adaptations include:

standardize constructor arguments;

remove repository-specific training logic;

return final logits.

# LS-FPN

The LS-FPN repository is dataset- and task-specific. To integrate it:

extract the model definition from the original project;

keep only the network forward pass;

adapt input and output channels;

return the final segmentation logits.

## nnU-Net

nnU-Net is not a single ordinary baseline model but a full self-configuring framework. It performs its own:

dataset conversion;

preprocessing;

planning;

training;

inference;

post-processing.

Therefore, we run nnU-Net using the official repository and compare its results externally.

## 3D U-Net

3D U-Net requires 3D inputs: [B, C, D, H, W]

The adaptations include:

modify the dataset loader to return 3D volumes;

modify transforms;

adapt evaluation to 3D outputs.

## SCUNet++

SCUNet++ relys on Swin Transformer blocks and CNN bottlenecks. The adaptations include::

installing model-specific dependencies;

setting image size and feature dimensions;

adapting input and output channels;

returning final logits only.

## UMamba

UMamba rely on Mamba-specific modules and custom configurations. The adaptations include:

installing the required Mamba-related dependencies;

extracting the segmentation network from the original project;

adapting input and output channels;

ensuring that the output has the same spatial size as the input;

returning final logits only.

## Citation
If this code or model is useful for your research, please cite:

Yiqi Wang, Jin Zhang, Ziqi Wang, Zhicai Peng, Xunliang Xu and Liang Zhao.
Bridging Unknown to Known: Connectivity-Guided Faithful Liver Vessel Segmentation, 2026.

## References
[1] Soler, L., Agnus, V., Fasquel, J., Moreau, J., Osswald, A., Bouhadjar, M., and Marescaux, J.
3D image reconstruction for comparison of algorithm database: A patient-specific anatomical and medical image database. Strasbourg, France, 2010.

[2] Simpson, A. L., Antonelli, M., Bakas, S., Bilello, M., Farahani, K., van Ginneken, B., Kopp-Schneider, A., Landman, B. A., Litjens, G., Menze, B., et al.
A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv:1902.09063, 2019.

[3] Gao, Z., Zong, Q., Wang, Y., Yan, Y., Wang, Y., Zhu, N., Zhang, J., Wang, Y., and Zhao, L.
Laplacian salience-gated feature pyramid network for accurate liver vessel segmentation. IEEE Transactions on Medical Imaging, 2023.
