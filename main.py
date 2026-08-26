from contextlib import asynccontextmanager
import pickle
import re
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# Model aur Tokenizer file paths
MODEL_PATH = "BiGRU_model (1).keras"
TOKENIZER_PATH = "tokenizer (1).pkl"

# Global dictionary for lifespan management
dl_model = {}

# 1. Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    dl_model["BiGRU"] = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Model and Tokenizer loaded successfully...")

    yield  # Server runs here and waits for requests

    dl_model.clear()  # Server stop hone par memory free kar dega when we press ctrl+c to stop the server
    print("Model cleared from memory.")


# 2. FastAPI App Setup
app = FastAPI(
    lifespan=lifespan
    )

# Constants
MAX_SEQUENCE_LENGTH = 50
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# 3. Text Preprocessing Function
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 4. Schemas
# Jo user input day ga
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )

# In what format response was generated
class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Enable CORS (Cross origin resourse sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"))

# Create API Endpoints

# Server UI at homepage ('/')
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

# Health Check Endpoint
@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

# Prediction Endpoint
@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input : TextInput):
    # clean the input sentences
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later")

    cleaned_text = preprocess_text(text_input.text)
    # Convert the text into tokens

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    # pad the sequence to make the same input length
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen = MAX_SEQUENCE_LENGTH,
        padding = "post",
        truncating = "post"
    )
    # run prediction using BiGRU model

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label : float(prob) for prob, label in zip(probabilities, EMOTION_LABELS)
    }
    # return the top emotion and full probability breakdown
    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = EMOTION_LABELS[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]),
        all_probabilities = all_probabilities
    )
    