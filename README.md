# Explainable WBC Classification

CPU-friendly five-class white blood-cell image classification using PyTorch,
MobileNetV3-Small or a custom CNN, Grad-CAM explanations, calibration metrics,
and a Streamlit interface.

## Dataset

Place the Raabin-WBC Double-Labeled Cropped WBC Images under `data/raw/` with
the official layout:

```text
data/raw/
  Train/Train/{Basophil,Eosinophil,Lymphocyte,Monocyte,Neutrophil}/
  TestA/TestA/{Basophil,Eosinophil,Lymphocyte,Monocyte,Neutrophil}/
```

`TestA` is kept as the held-out five-class test split. `TestB` is not included
in five-class evaluation because it contains only lymphocytes and neutrophils.
The generated manifest and reports are written under `data/processed/` and
`results/`.

## Environment setup

Windows PowerShell:

```powershell
conda env create -f environment.yml
conda activate wbc-ai
```

If the environment already exists, install/update the dependencies with:

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" -m pip install -r requirements.txt
```

## Train and evaluate

Run a short CPU smoke test first:

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" src/train.py --model mobilenet_v3_small --subset 0.03 --epochs 2 --batch_size 16 --image_size 128
```

For the recommended accuracy run, omit `--subset` so every training image is
used. The default is now MobileNetV3-Small with ImageNet transfer learning,
because the saved baseline CNN has a substantially lower held-out macro-F1.
Checkpoints are saved in `artifacts/`, and training history/curves are saved in
`results/`. Training can resume from a latest checkpoint with `--resume`.

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" src/evaluate.py --checkpoint artifacts/mobilenet_v3_small_best.pt --image_size 128
```

## Applications

Start the Streamlit application:

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" -m streamlit run app/streamlit_app.py
```

Start the optional FastAPI service:

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API exposes `/health`, `/model/info`, and multipart-image `POST /predict`.
It returns model scores and a Base64 Grad-CAM overlay when a checkpoint exists.

## Tests

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" -m pytest tests -q
```

The application is an educational research prototype, not a medical device.
Softmax scores and Grad-CAM regions do not establish clinical certainty or
biological causation.

## Latest verified results

The recommended MobileNetV3-Small checkpoint was evaluated on all 4,339 held-out
TestA images:

- Accuracy: **91.31%**
- Macro-F1: **0.8761**
- Macro ROC-AUC: **0.9914**
- Average CPU latency: **2.62 ms/image**

The modernized CellScope AI interface shows the top prediction, runner-up class,
confidence margin, Grad-CAM evidence and close-call warnings. The baseline CNN
remains available for comparison.
"# WBC-classification" 
