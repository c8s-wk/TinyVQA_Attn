# TinyVQA_Attn
A lightweight Visual Question Answering (VQA) system that combines image understanding (CNN) and question processing (bag-of-words / dense layers) with multimodal attention to predict answers.

## Dataset and reference
EasyVQA 
https://github.com/vzhou842/easy-VQA.git

- Change the fusion (element-wise multiplication) with transformer
- Use question embedding as query and image features as key/value.
- Apply MultiHeadAttention (from Keras)

## Requirement
### Set up venv (Optional)
```bash
python -m venv .venv

# Activate
.venv/Scripts/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Training
Train the model with:
```bash
python train.py
```
This will:
- Build the multimodal attention model
- Train on *data/train*
- Save best model to *files/model.h5*
- Log training metrics to *files/data.csv*

## Evaluation
Evaluate on the test set:
```bash
python evaluate.py
```
This will:
- Load *files/model.h5*
- Predict answers on *data/test*
- Print a classification report 
- Save a confusion matrix heatmap