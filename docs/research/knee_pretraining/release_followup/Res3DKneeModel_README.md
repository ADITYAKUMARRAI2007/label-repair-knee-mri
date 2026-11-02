# Res3DKneeModel

![Model Architecture](images/model.png)

## Description
This repository contains a 3D Residual Network model for knee MRI analysis. The model processes multi-sequence MRI data to predict various knee pathologies using a combined 3D convolutional approach with transformer-based feature aggregation.

## Requirements
All required dependencies are listed in `requirements.txt`.

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Model Architecture
The model architecture combines a ResNet-18 backbone with 3D convolutions that operate across slices of multi-sequence knee MRI data. It uses a transformer encoder to aggregate features across slices and sequences. Key components include:
- `Res3Dish`: Base feature extractor with 3D convolutions
- `SliceTransformer`: Combines ResNet features with transformer encoding

## Training
Training is performed using Slurm workload manager. The `train.sh` script handles the training process:

```bash
sbatch train.sh
```

### Training Parameters
In `train.sh`:
- `FOLD`: Specifies which fold to use for cross-validation
- `EXPERIMENT`: Name of the experiment/pathology
- `PY_ARGS`: Class index to train for (pathology identifier)

## Dataset
The model works with knee MRI data in three planes (coronal, sagittal, transversal) plus an optional T1 sequence. The data needs to be preprocessed and normalized before training.
