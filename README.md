# DeepBasis_SVBRDF
This is the source code of research paper "DeepBasis: Hand-Held Single-Image SVBRDF Capture via Two-Level Basis Material Model" (Proceedings of SIGGRAPH Asia 2023).
**More information (include our paper, supplementary, video) can be found at** [My Personal Page](https://www.baidu.com) 
![Alt](Teaser2.jpg)

# Pretrained models
Our pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1t7nnzP2htXwPQVajYdHqlv-MyN_zYPBb?usp=drive_link). Download them and extract them into ```./model/ ```

# Dependencies
- Python (with opencv-python, numpy; Python tested on 3.7)
- Pytorch (tested on 1.10.1+CUDA 11.1)

# Usage
- Train mode
```Python
python train.py
  --name DeepBasisTraining   # experiment name.
  --save_root ./output # output folder.
  --train_data_root ./dataset/train # training data folder.
  --test_data_root ./dataset/test # test data folder during training.
  --use_tb_logger # whehter to use tensorboard logger.
  --fovZ 4 # Acording to FOV, compute the z-coordinate.
  --total_iter 400000
  --print_freq 100
  --save_freq 100000
  --test_freq 100000
```

- Test mode (synthetic data as input for comparison)
```Python
python test.py
  --name DeepBasisTest # experiment name.
  --save_root ./output # output folder.
  --test_data_root ./source/test # test svbrdf data folder
  --loadpath_network_g ./pretrain/net_g_2.414.pth
  --loadpath_network_l ./pretrain/net_l_2.414.pth
  --fovZ 2.414 # Acording to FOV, compute the z-coordinate.
```
- Real mode (real-captured images as input)
```Python
python real.py
  --name DeepBasisRealCapture # experiment name.
  --save_root ./output # output folder.
  --real_data_root ./source/real # folder for images
  --loadpath_network_g ./pretrain/net_g_4.pth
  --loadpath_network_l ./pretrain/net_l_4.pth
  --fovZ 4
```
Note that each pretrained model corresponds to a fixed FOV (Field of View), and during testing, it's essential to maintain the alignment between the captured image's FOV and the model. Specifically, when the captured image undergoes cropping before input (e.g., checkboard capture), it should ensure the correspondence of the actual FOV after cropping.
# Citation
If you use our code or pretrained models, please cite as following:
```

```
