# Denoising IMU Data with Transformers
### Overview
This repository provides a general framework for IMU denoising.
The framework allows for selecting from different types of denoising methods and different types of backbones.

We currently support the following denoising methods
- Sequence-to-sequence
- Residual-based

We currenty support the following backbones:
- CNNB: a cnn architecture, with the same structure as the cnn used in Brossard et al 2020
- Transformer: a transformer-based architetcure 

### Training 
```main.py``` is the entry point for the framework. Run as follows for training:
```python main.py --mode train-val --train_imu_dataset_file sample_data/07_03_21_static_random_accel_100_seq_of_1k_constant_train.csv --val_imu_dataset_file sample_data/07_03_21_static_random_accel_20_seq_of_1k_constant_test_1.csv --config_path configs/cnnb_residual.json```
You can use other configuration files to train with different methods of denoising or with different backbones.
See also ```example_args.txt``` for more example arguments

### Testing
You can validation alone or after training. See examples at ```example_args.txt```