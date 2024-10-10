# DMatch
This repo is the official implementation for DMatch: Distinguish Confusing Classes with Unlabeled Data for Semi-Supervised Learning


## Architecture of DMatch     
![image](./images/Architecture.png)

# Prerequisites


- Python >= 3.6
- PyTorch >= 1.6.0
- torchvision >= 0.7.0
- matplotlib
- numpy
- pillow

- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e torchlight` 


# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- CIFAR-10/100   
- SVHN    


### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - cifar-10-batches-py/    # from 'cifar-10-python.tar.gz'
  - cifar-100-python/       # from 'cifar-100-python.tar.gz'
    ...
    ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training

- Change the config file depending on what you want, and you can refer to the [script folder](https://github.com/OliverHxh/SkeletonGCL/tree/main/script) for the more examples. Notably, please just use one GPU for training because we find that using mutiple GPUs would affect the performances.

```
# Example: training SkeletonGCL with CTRGCN on NTU RGB+D 120 cross-subject with joint modality on GPU 0
python main.py --config config/nturgbd120-cross-subject/ctr.yaml --work-dir work_dir/ntu120/csub/ctrgcn_joint --device 0
```
```
# Example: training SkeletonGCL with 2s-AGCN on NTU RGB+D 120 cross-subject with joint modality on GPU 0
python main.py --config config/nturgbd120-cross-subject/agcn.yaml --work-dir work_dir/ntu120/csub/agcn_joint --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training SkeletonGCL with CTRGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/ctr.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn_bone --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/xxx.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of CTRGCN on NTU RGB+D 120 cross subject
python ensemble.py --dataset ntu120/xsub \
--joint-dir work_dir/ntu120/xsub/ctrgcn_MGGCL_joint \
--bone-dir work_dir/ntu120/xsub/ctrgcn_MGGCL_bone \
--joint-motion-dir work_dir/ntu120/xsub/ctrgcn_MG_joint_motion \
--bone-motion-dir work_dir/ntu120/xsub/ctrgcn_MG_bone_motion
```
