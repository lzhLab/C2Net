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

#### 5. Proposed Model: C<sup>2</sup>Net
The proposed model is implemented in:

models/C2Net.py
The default callable wrapper is:

from models.C2Net import RNN_Model
The model follows a U-shaped architecture:

Encoder: continuity-guided convolutional residual units

Encoder channel widths: 32, 64, 128, 256

Decoder: atrous convolution blocks

Atrous convolution setting: kernel_size=3, padding=2, dilation=2

Normalization: BatchNorm

Activation: ReLU

A simplified model construction example:

from models.C2Net import RNN_Model

model = RNN_Model(
    input_channels=3,
    output_channels=1,
    base_channels=32
)
The default input channel number is controlled by:

input_channels = expand_size * 2 + 1
For example, if expand_size=1, the model input has 3 channels.

6. Supported and Optional Baseline Models
The training script is designed to support multiple baseline segmentation models through a unified interface.

No.	Model	Reference	Code Link
1	U-Net	Ronneberger et al., MICCAI 2015	https://github.com/milesial/Pytorch-UNet
2	PSPNet	Luo et al., Engineering Letters 2024	https://github.com/hszhao/semseg
3	TransUNet	Chen et al., arXiv 2021	https://github.com/Beckschen/TransUNet
4	SA-UNet	Guo et al., ICPR 2021	https://github.com/clguo/SA-UNet
5	R2U-Net	Alom et al., arXiv 2018	https://github.com/LeeJunHyun/Image_Segmentation
6	RU-Net	Wang et al., CMPB 2022	Project-specific implementation
7	LS-FPN	Gao et al., IEEE TMI 2023	Project-specific implementation
8	nnU-Net	Isensee et al., Nature Methods 2020	https://github.com/MIC-DKFZ/nnUNet
9	3D U-Net	Huang et al., Computers in Biology and Medicine 2018	https://github.com/wolny/pytorch-3dunet
10	SCUNet++	Chen et al., WACV 2024	Project-specific implementation
11	UMamba	Jain et al., BSPC 2026	Project-specific implementation

7. Unified Model Interface
To make a model callable from train_seg.py, each model should follow the same interface.

7.1 Recommended Constructor
class YourModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32):
        super().__init__()
        ...
or:

class YourModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, base_channels: int = 32):
        super().__init__()
        ...
7.2 Recommended Forward Function
For 2D models:

def forward(self, x):
    """
    Args:
        x: Tensor with shape [B, C, H, W]

    Returns:
        logits: Tensor with shape [B, out_channels, H, W]
    """
    return logits
For 3D models:

def forward(self, x):
    """
    Args:
        x: Tensor with shape [B, C, D, H, W]

    Returns:
        logits: Tensor with shape [B, out_channels, D, H, W]
    """
    return logits
7.3 Required Output
All models should return logits, not probabilities:

return logits
The loss function applies sigmoid internally when needed.

8. Adding a New Baseline Model
To add a new model, follow these steps.

Step 1: Add the Model File
Example for PSPNet:

models/
└── pspnet/
    ├── __init__.py
    └── pspnet.py
Step 2: Add a Wrapper If Needed
Many official implementations use different argument names, such as num_classes, n_channels, or n_classes. Wrap them into the unified interface:

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
Step 3: Register the Model in models/__init__.py
from .unet import Unet
from .C2Net import RNN_Model
from .pspnet.pspnet import PSPNet
Step 4: Add the Model to train_seg.py
Add it to the model registry or import list, depending on your version of train_seg.py:

from models.pspnet.pspnet import PSPNet
Then ensure the architecture name is supported:

choices=[
    "Unet",
    "RNN_Model",
    "PSPNet",
    ...
]
Run:

python train_seg.py --arch PSPNet
9. Model Registry in train_seg.py
The recommended minimal registration for the current project is:

from models import Unet
from models.C2Net import RNN_Model

MODEL_REGISTRY = {
    "Unet": Unet,
    "RNN_Model": RNN_Model,
}
Optional models can be added as follows:

from models.pspnet.pspnet import PSPNet
from models.transunet.transunet import TransUNet
from models.sa_unet.sa_unet import SAUNet
from models.r2_unet.r2_unet import R2UNet
from models.ru_net.ru_net import RUNet
from models.ls_fpn.ls_fpn import LSFPN
from models.unet3d.unet3d import UNet3D
from models.scunet_pp.scunet_pp import SCUNetPP
from models.umamba.umamba import UMamba

MODEL_REGISTRY.update({
    "PSPNet": PSPNet,
    "TransUNet": TransUNet,
    "SAUNet": SAUNet,
    "R2UNet": R2UNet,
    "RU_Net": RUNet,
    "LSFPN": LSFPN,
    "UNet3D": UNet3D,
    "SCUNetPP": SCUNetPP,
    "UMamba": UMamba,
})
If a model is not registered, running:

python train_seg.py --arch ModelName
will raise an error similar to:

ImportError: Model 'ModelName' is not available.
10. Model-Specific Notes
U-Net
U-Net is the simplest 2D baseline. It should work directly if models/unet.py defines:

class Unet(nn.Module):
    ...
Run:

python train_seg.py --arch Unet
PSPNet
PSPNet is originally designed for natural-image semantic segmentation. To use it here:

Set in_channels to expand_size * 2 + 1

Set the final classifier output to output_size

Ensure output resolution is upsampled to the input resolution

Return only the final logits

Run:

python train_seg.py --arch PSPNet
TransUNet
TransUNet usually requires:

Image size configuration

ViT patch size configuration

Pretrained ViT weights, if used

Final output channel modification

For this project, wrap it so that:

TransUNet(in_channels, out_channels, base_channels)
returns logits with shape:

[B, out_channels, H, W]
Run:

python train_seg.py --arch TransUNet --batch-size 4 --lr 1e-4
SA-UNet
SA-UNet can be treated as a 2D U-Net variant. Required changes:

Match input channels

Match output channels

Return logits only

Run:

python train_seg.py --arch SAUNet
R2U-Net
R2U-Net commonly uses recurrent residual blocks with t=2.

Required changes:

Match input channels

Match output channels

Set recurrent depth if needed

Return logits only

Run:

python train_seg.py --arch R2UNet
RU-Net
RU-Net implementations vary across repositories. Required changes:

Wrap the model constructor

Match input channels

Match output channels

Return logits only

Run:

python train_seg.py --arch RU_Net
LS-FPN
LS-FPN is a liver vessel-specific segmentation baseline. Required changes:

Add the model implementation under models/ls_fpn/

Wrap the constructor

Match input and output channels

Return final segmentation logits only

Run:

python train_seg.py --arch LSFPN
nnU-Net
nnU-Net is a self-configuring framework and is best used through its official pipeline.

Recommended usage:

Convert your dataset to the nnU-Net format.

Run nnU-Net preprocessing and planning.

Train and evaluate using the official nnU-Net commands.

If you still want to call an nnU-Net-like network inside train_seg.py, provide a wrapper class under:

models/nnunet/nnunet.py
Run:

python train_seg.py --arch nnUNet
3D U-Net
3D U-Net requires 5D input tensors:

[B, C, D, H, W]
The current train_seg.py and dataset pipeline are primarily designed for 2D slice-based training. To run 3D U-Net, you must modify the dataset loader so that it returns volumetric or stacked-slice inputs.

Run only after adapting the dataset:

python train_seg.py --arch UNet3D
SCUNet++
SCUNet++ may require additional dependencies such as timm.

Required changes:

Wrap the constructor

Match input channels

Match output channels

Return logits only

Run:

python train_seg.py --arch SCUNetPP
UMamba
UMamba implementations may depend on Mamba-specific packages.

Required changes:

Install model-specific dependencies

Wrap the constructor

Match input channels

Match output channels

Return logits only

Run:

python train_seg.py --arch UMamba

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
