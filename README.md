# KFBI-DETR: A forestry pest detection model based on multi-frequency features and KAN convolution
- This is an official implementations for the paper "A forestry pest detection model based on multi-frequency features and KAN convolution"
- KFBI-DETR takes Deformable-DETR as the basic framework, uses ResNet50 as the backbone and introduces the KFBIFPN (Frequency KAN BIFPN) module.

## Installtion
### Requirements

* Linux, CUDA>=11.3, GCC>=5.4
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n kfbi_detr python=3.8
    ```
    Then, activate the environment:
    ```bash
    conda activate kfbi_detr
    ```
  
* PyTorch>=1.12.1, torchvision>=0.13.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 11.3, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```

or without using bash command:

```bash
python main.py --pretrain_model_path path/to/your/pretrained.pth
```

#### Training on multiple nodes

For example, the command for training Deformable DETR on 2 nodes of each with 8 GPUs is as following:

On node 1:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

On node 2:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 8 configs/r50_deformable_detr.sh
```

Or 2 nodes of  each with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 16 configs/r50_deformable_detr.sh
```
#### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 1' or '--batch_size 2'.

### Evaluation

You can get the config file and pretrained model of KFBI-DETR (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 validation set:

```bash
python main.py --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh``` or ```./tools/run_dist_slurm.sh```.
