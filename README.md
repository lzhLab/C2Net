# C<sup>2</sup>Net

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

Parameters are editable in `train_seg.py`, the connectivity is pretrained by using `models/mgu.py`, and the `pth` file was stored in `models/pretrained`.

## Datasets

The datasets used for model training and evalutation are 3DIRCADb[1], MSD[2], and LiVS[3].
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

## Reference
[1] Soler L, H.A.，Agnus V,C.A.,Fasquel J，M.J.，Osswald A，B.M.，J, M.: 3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database. Strasbourg,France (2010) \par
[2] Simpson，A.L.，Antonelli,M.，Bakas,S.,Bilello，M.，Farahani，K., Van Ginneken, B.，Kopp-Schneider，A.,，Landman，B.A.，Litjens， G.,Menze, B., et al.: A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv:1902.09063(2019) \par
[3] Gao, Z.,Zong, Q.,Wang,Y.， Yan，Y.,Wang，Y., Zhu， N., Zhang,J.,
Wang,Y., Zhao,L.: Laplacian salience-gated feature pyramid network for accurate liver vessel segmentation. IEEE Transactions on Medical Imaging (2023)
