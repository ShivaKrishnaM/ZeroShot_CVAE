# CVAE_ZSL

You need keras with tensorflow backend to run this code. To train the SVM model, you also need sklearn

Download the Data

* You can download the CUB data from [this link](https://drive.google.com/drive/folders/1zpG3vElAeIGUzPWwgGanLzKbV7a68cNS?usp=sharing)

* Unzip the file and place the 'CUB' folder in Datasets/

To partition the data into test and train classes
```
cd Disjoint/CUB/
python testTrainSplit.py
```
To train the CVAE model and run the subsequent supervised SVM classifier. 
```
python trainCVAE.py
```

If you find our code useful, please cite our work: 
```
@InProceedings{Mishra_2018_CVPR_Workshops,
author = {Mishra, Ashish and Krishna Reddy, Shiva and Mittal, Anurag and Murthy, Hema A.},
title = {A Generative Model for Zero Shot Learning Using Conditional Variational Autoencoders},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
} 
```
