from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

class InputText(BaseModel):
    text: str

# ✅ Base model sentiment
def base_model_sentiment(text: str):
    result = sentiment_pipeline(text)[0]
    sentiment = "positive" if "POS" in result["label"].upper() else "negative"
    confidence = round(result["score"] * 100, 2)
    return sentiment, confidence

# ✅ Human-like reasoning layer
def human_like_verifier(text: str, predicted_sentiment: str, confidence: float):
    text_lower = text.lower()

    negation_words = ["not", "never", "no", "none", "n't"]
    contrast_words = ["but", "however", "although", "though", "yet", "nevertheless", "still"]
    neutral_words = ["okay", "fine", "average", "decent", "moderate", "neutral", "normal"]
    negative_cues = [
        "worse", "not good", "poor", "bad", "free of cost", "better than this",
        "less feature", "slow", "lag", "issue", "problem"
    ]

    # 🧠 Rule 1: Negation before positive
    if any(word in text_lower for word in negation_words):
        if predicted_sentiment == "positive":
            predicted_sentiment = "negative"
            confidence = min(confidence + 10, 99.9)

    # 🧠 Rule 2: Handle contrast — weigh part after "but"/"however"
    for c_word in contrast_words:
        if f" {c_word} " in text_lower:
            parts = text_lower.split(f" {c_word} ", 1)
            after = parts[-1]
            after_result = sentiment_pipeline(after)[0]
            final_label = after_result["label"]
            predicted_sentiment = "positive" if "POS" in final_label.upper() else "negative"
            confidence = round(after_result["score"] * 100, 2)
            break

    # 🧠 Rule 3: Check for negative comparative cues
    if any(word in text_lower for word in negative_cues):
        predicted_sentiment = "negative"
        confidence = max(confidence, 80)

    # 🧠 Rule 4: Weak language → neutralize
    if any(word in text_lower for word in neutral_words):
        predicted_sentiment = "neutral"
        confidence = round(confidence * 0.8, 2)

    # 🧠 Rule 5: Low confidence → neutral
    if confidence < 50:
        predicted_sentiment = "neutral"

    return predicted_sentiment, confidence


@app.post("/predict")
def predict_sentiment(data: InputText):
    model_sentiment, model_confidence = base_model_sentiment(data.text)
    final_sentiment, final_confidence = human_like_verifier(
        data.text, model_sentiment, model_confidence
    )
    return {"Sentiment": final_sentiment, "Confidence": final_confidence}
