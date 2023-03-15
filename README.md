# C2Net

Code for paper:"Bridging unknown to known: connectivity-guided faithful liver vessel segmentation" by Yanxin Ye et al.

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

You can change parameters in `train_seg.py`. We pretrained  `models/gru.py` using the training part of the datasets and the `pth` file was stored in `models/pretrained`.

## Datasets

We use 3DIRCADb, MSD, LiVS datasets.

Images and GT_labels should be organized as follows:

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

```
