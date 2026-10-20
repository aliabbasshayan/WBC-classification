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
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" src/train.py --model baseline_cnn --subset 0.03 --epochs 2 --batch_size 16 --image_size 128
```

For a longer run, omit `--subset`. Checkpoints are saved in `artifacts/`, and
training history/curves are saved in `results/`. Training can resume from a
latest checkpoint with `--resume`.

```powershell
& "$env:USERPROFILE\anaconda3\envs\wbc-ai\python.exe" src/evaluate.py --checkpoint artifacts/baseline_cnn_best.pt --image_size 128
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
"# WBC-classification" 
