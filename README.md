# IMDiffusion

This repository is the implementation of IMDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection. We propose the IMDiffusion framework for unsupervised anomaly detection and evaluate its performance on six open-source datasets.

# Results
The main results are presented in the following table. Our method outperforms the previous unsupervised anomaly detection methods in the majority of metrics.

![Image Description](result.png)

# Data
To use the data mentioned in our paper, firstly download Machine.zip from https://drive.google.com/file/d/1VX5P60gS6fIJ_XDvnlF6M5fwJbWxwkNI/view?usp=sharing and put it in /data.
Please unzip Machine.zip in data. The data of SWaT has not been uploaded, and you need to apply by its official tutorial.

To use new dataset, please:

1. upload {dataset name}_train.pkl, {dataset name}_test.pkl, {dataset_name}_test_label.pkl
2. You need to add some code in exe_machine.py. If your dataset contains multiple sub datasets, you can refer to the practice of registering the names of each sub dataset in the SMD dataset. Meanwhile, please add its feature_dim parameter.
3. Please add feature_dim parameter in evaluate_machine_window_middle.py

# Train and inference


To reproduce the results mentioned in our paper, first, make sure you have torch and pyyaml installed in your environment. Then, use the following command to train:
```shell
python exe_machine.py --device cuda:0 --dataset SMD
```

To obtain the average performance of the model, we run six times for one dataset. You can modify this parameter according to your needs.

After completing the training, you can use the

```shell
python evaluate_machine_window_middle.py --device cuda:0 --dataset SMD
```

command to perform inference.

After inference, if you want to compute score, there are two scripts for compute score: compute_score.py and ensemble_proper.py.

compute_score.py is used to calculate the score of each dataset at each threshold. For individual data such as SWaT, SMAP, MSL, and PSM that do not contain sub datasets, you do not need to run this code.

If you want to compute score on SMD and GCP, firstly 
```shell
python compute_score.py --dataset_name SMD
```

then

```shell
python ensemble_proper.py --dataset_name SMD
```

else if you want to compute score for PSM, SWaT, MSL and SMAP, just use:

```shell
python ensemble_proper.py --dataset_name SWaT
```

you can check the result in ensemble_residual, we use a csv file to record average score, for example:

| average    |            |            |            |
| ---------- | ---------- | ---------- | ---------- |
| p          | r          | f1         | add        |
| 0.91759533 | 0.84315562 | 0.87865043 | 324.476196 |
| std        |            |            |            |
| p          | r          | f1         | add        |
| 0.01279164 | 0.01456169 | 0.00622323 | 17.4200802 |


# Cite this work

If you use this work, please cite using the following bibtex entry.

```markdown
@article{chen2023imdiffusion,
title={ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection},
author={Chen, Yuhang and Zhang, Chaoyun and Ma, Minghua and Liu, Yudong and Ding, Ruomeng and Li, Bowen and He, Shilin and Rajmohan, Saravan and Lin, Qingwei and Zhang, Dongmei},
journal={arXiv preprint arXiv:2307.00754},
year={2023}
}
```
🚀 Advanced Setup Guide (Python 3.14 & Custom Data)
This guide documents how to run IMDiffusion on the latest Python 3.14 using nightly PyTorch builds with CUDA 12.6 support, and how to train it on your own datasets.

1. Environment Setup (The Bleeding Edge Way)
Standard installations will fail on Python 3.14. You must use the nightly repository to get a GPU-enabled build.

A. Clean Installation
Open your terminal in the project folder:

PowerShell

# 1. Create and activate a clean virtual environment
py -m venv venv
.\venv\Scripts\activate

# 2. IMPORTANT: Remove any cached CPU versions if you previously tried installing
py -m pip uninstall torch torchvision torchaudio -y
py -m pip cache purge

# 3. Install PyTorch Nightly with CUDA 12.6 Support
# This specific command targets the experimental build compatible with Python 3.14
py -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 4. Install remaining dependencies
py -m pip install numpy pyyaml pandas gdown
Verification: When running step 3, ensure the download size is > 2.0 GB. If it is small (~110MB), you are getting the CPU version.

2. Training on Custom Data
To use your own data instead of the academic datasets (SMD/PSM), follow this workflow.

Step A: Prepare Your Data
You need three CSV files:

train.csv (Training data, columns = features)

test.csv (Test data, same columns)

test_label.csv (Binary labels: 0=Normal, 1=Anomaly)

Step B: Convert to .pkl Format
Create a file named convert_data.py in the root directory and paste the code below. This converts your CSVs into the specific NumPy format the model requires.

File: convert_data.py

Python

import numpy as np
import pandas as pd
import pickle
import os

# --- CONFIGURATION ---
DATASET_NAME = "MyData"   # Name used in command line later
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
LABEL_CSV = "test_label.csv"
# ---------------------

def create_dataset():
    output_dir = os.path.join("data", "Machine")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {DATASET_NAME}...")

    try:
        # Load CSVs (Assumes no header row. If you have headers, add header=0)
        train_data = pd.read_csv(TRAIN_CSV, header=None).values.astype(np.float32)
        test_data = pd.read_csv(TEST_CSV, header=None).values.astype(np.float32)
        test_label = pd.read_csv(LABEL_CSV, header=None).values.flatten().astype(np.float32)
    except FileNotFoundError:
        print("❌ Error: CSV files not found. Please ensure train.csv, test.csv, and test_label.csv exist.")
        return

    print(f"Feature Dimension: {train_data.shape[1]}")
    
    # Save as .pkl
    with open(os.path.join(output_dir, f"{DATASET_NAME}_train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, f"{DATASET_NAME}_test.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    with open(os.path.join(output_dir, f"{DATASET_NAME}_test_label.pkl"), "wb") as f:
        pickle.dump(test_label, f)
    
    print("✅ Conversion complete! Files saved to data/Machine/")

if __name__ == "__main__":
    create_dataset()
Run the converter:

PowerShell

py convert_data.py
Step C: Register Dataset in Code
Open exe_machine.py and modify the dataset selection logic (approx. line 118). You must tell the model how many features (columns) your data has.

Python

    if args.dataset == "SMD":
        feature_dim = 38
    elif args.dataset == "MyData":   # <--- ADD THIS BLOCK
        feature_dim = 5              # <--- Set this to YOUR number of columns
    elif args.dataset == "SMAP" or args.dataset == "PSM":
        feature_dim = 25
3. Running the Model
Train
PowerShell

py exe_machine.py --dataset MyData --device cuda:0
Evaluate
PowerShell

py evaluate_machine_window_middle.py --dataset MyData --device cuda:0
4. Common Fixes
If you encounter errors during execution, apply these code fixes:

FileExistsError in exe_machine.py:

Find: os.makedirs(foldername)

Replace with: os.makedirs(foldername, exist_ok=True)

Indentation Errors:

Ensure your elif args.dataset == "MyData": block is aligned exactly with the if statement above it.
