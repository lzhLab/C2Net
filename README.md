# C^2Net

Codes for the manuscript: "Bridging unknown to known: connectivity-guided faithful liver vessel segmentation" by Ye et al.

## Dependencies

```
python 3.8
pytorch = 1.8
```

## Usage

```
Run the main program:     
	python train_seg.py
```

Parameters are editable in `train_seg.py`, the connectivity is pretrained by using `models/gru.py`, and the `pth` file was stored in `models/pretrained`.

## Datasets

The datasets used for model training and evalutation are 3DIRCADb, MSD, and LiVS.
The input images and groundtruth labels should be organized as follows:

```
├── datasets
|   ├── data
|   	├── LiVS
|   		├── image
|   			├── 01
|   				├── 01_1.png
|   				├── 01_2.png
|   				......
|   			├── 02
				......
|   		├── label
|   			├── 01
|   				├── 01_1.png
|   				├── 01_2.png
|   				....
|   			├── 02
				......
|   	├── MSD
		......
|   	├── 3DIRCADb
		......
```

## Citation

If the model is useful for your research, please cite:

```
Yanxin Ye, Xu Xiao, Jin Zhang, Ruonan Wu, Ziqin Huang, Jie Luo, and Liang Zhao. Bridging unknown to known: connectivity-guided faithful liver vessel segmentation, 2023.
```
