# download_model.py
from transformers import pipeline
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
print("Downloading model:", MODEL)
pipeline("sentiment-analysis", model=MODEL)
print("Model downloaded.")