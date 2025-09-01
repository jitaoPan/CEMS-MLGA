## CEMS & MLGA

This is the open-source code for the paper "Class-Enhanced Multi-Sampling and Multi-Level Graph Attention for 3D Object Detection."

![img.png](img.png)

This project is based on OpenPCDet. So please refer to INSTALL.md and GETTING_STARTED.md for the installation and usage, respectively。


Since the original networks were scattered across different OpenPCDet projects, we are still in the process of integrating all the code. Thank you for your patience.

If you want to try our reproduced IA-SSD, please run the following commands after setting up the environment.
```
cd tools
python train.py --cfg_file cfgs/once_models/IA-SSD-ctr_aware.yaml 
```
If you wish to try IA-SSD with CEMS and MLGA modules, please use the following commands:
```
cd tools
python train.py --cfg_file cfgs/once_models/CA-SSD_cams.yaml 
```

[//]: # (The voxel-based code is currently being integrated into the unified project, as it was adapted from the SAFDNet open-source implementation. Please stay tuned for updates.)

If you want to try origin SECOND, please run the following commands after setting up the environment.
```
cd tools
python train.py --cfg_file cfgs/once_models/second.yaml 
```

If you want to try SECOND with CEMS and MLGA modules, please run the following commands after setting up the environment.
```
cd tools
python train.py --cfg_file cfgs/once_models/second_with_cems_mlga.yaml 
```


If you want to try origin SAFDNet, please run the following commands after setting up the environment.
```
cd tools
python train.py --cfg_file cfgs/waymo_models/safdnet.yaml
```

If you want to try SAFDNet Backbone that has add CEMS, please run the following commands after setting up the environment.
```
cd tools
python train.py --cfg_file cfgs/waymo_models/safdnet_backbone_with_cems.yaml
```

**Note: You must additionally install torch.scatter according to your PyTorch and CUDA versions to run SAFDNet-related code.**

If you want to use the CEMS and MLGA modules in other networks, please refer to the module configuration methods in these two configuration files: second_cems_mlga.yaml and CA-SSD_cams.yaml.



