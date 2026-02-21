# deep_learning_emg
# EMG Signal Classification with Superlet + TCN

## Description
This project analyzes EMG (electromyography) signals using **Superlet feature extraction** and a **Temporal Convolutional Network (TCN)** with Squeeze-and-Excitation (SE) attention.  
The goal is to classify muscle movements accurately and efficiently, providing a foundation for biomedical signal analysis and AI-driven healthcare solutions.

## Dataset
The EMG data used in this project is **freely available** from [Open Data Platform link](https://ninapro.hevs.ch/instructions/DB1.html).  

After downloading the `.mat` files, you can generate the feature arrays (`.npz`) required for training by running:

python intrasubject_superlet_TCN_v3.py

This will create the superlet_features_subject1_400ms_30freqs.npz file automatically.

## Installation

It is recommended to create a Python virtual environment:
`
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
`
Install the required packages:
`
pip install -r requirements.txt
`
## Usage

Download the raw .mat EMG data.

Run the script to extract features and train the model:
`
python intrasubject_superlet_TCN_v3.py
`
The script will generate the following outputs:

best_tcn_superlet.pt → Trained model weights

loss_curve.png → Training and validation loss plot

accuracy_curve.png → Training and validation accuracy plot

confusion_matrix.png → Confusion matrix on the validation set

## Requirements

Python >= 3.10

numpy

scipy

torch

scikit-learn

matplotlib

(Exact versions are listed in requirements.txt.)

## Notes

- The .npz feature file is generated from the raw .mat data. Do not upload .npz to GitHub if the raw data is not yours.

- This project is intended for educational and research purposes only.

- Make sure to have a GPU available for faster training, although CPU training is supported.












