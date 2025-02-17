# IGD_Vehicle_Exterior_Shape

## Authors:
- Yuhao Liu  
- Maolin Yang  
- Pingyu Jiang  
- All authors are affiliated with the State Key Laboratory of Mechanical Manufacturing Systems 
- at Xi'an Jiaotong University.
---

## Code Usage Instructions:

### 1. Python Version:
Python 3.10  

### 2. Install Required Libraries:
Install the required libraries using the following command:   
```bash
pip install -r requirements.txt
```
### 3. Folder Structure and Contents:
The project contains the following folders and files:
a) checkpoints folder:
Contains 4 trained models for vehicle exterior shape generation.
b) models_3D_obj folder:
Vehicle exterior 3D dataset in .obj format.
c) models_labels_npy folder:
Vehicle label files, input conditions processed as 2048x3 in .npy format.
d) model_pointcloud_npy folder:
Vehicle point cloud data, sampled at 2048x3 in .npy format.
### 4. Scripts:

a) Training Script:
Run the following command to train the model:
```bash
python run train_improved_cgan.py
```
b) Conditional Generation Script:
Run the following command for conditional generation:
```bash
python run test_improved_cgan.py
```