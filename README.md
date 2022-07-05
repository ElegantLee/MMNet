# MMNet
A PyTorch implementation of "MMNet: A medical image-to-image translation network based on manifold value correction and manifold matching".

![MMNet](https://typora-lee.oss-cn-chengdu.aliyuncs.com/img-typora/MMNet.png)

---
## Environment
1. Linux(ubuntu)
2. python=3.7.3
3. torch=1.7.1+cu92
4. tqdm=4.32.1
5. opencv-python=4.5.3.56
6. pyyaml=5.1.1
7. visdom=0.1.8.9

---

## Demo

1. The pre-trained models are in "checkpoints/MMNet";

2. Download patial test samples from [GoogleDrive](https://drive.google.com/drive/folders/1C6XNJSUw_1kR8fwYot47WlJTb08pFRu0?usp=sharing), then put them into corresponding dir ("datasets/BraTs2015/val or datasets/OASIS3/val");

3. Modify the MMNet.yaml in "Yaml/MMNet.yaml";

   Config for test.

   ```js
   run_name: 'MMNet/BraTs2015/'
   dataset: BraTs2015
   val_dataroot: 'datasets/BraTs2015/val'
   input_nc: 1
   ```

   ```js
   run_name: 'MMNet/OASIS3/'
   dataset: OASIS3
   val_dataroot: 'datasets/OASIS3/val'
   input_nc: 3
   ```

   

4. ```js
   python test.py
   ```

---

## Acknowledgments

Code borrows from [Reg-GAN](https://github.com/Kid-Liet/Reg-GAN) and [pytorch-manifold-matching](https://github.com/dzld00/pytorch-manifold-matching). The distribution generator and distribution corrector is modified from [Reg-GAN](https://github.com/Kid-Liet/Reg-GAN) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
