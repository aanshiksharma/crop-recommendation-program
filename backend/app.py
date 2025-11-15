from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch import nn
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Helper Functions from aicrop.py ---

# Mock profit table from aicrop.py
profit_table = {
    "rice": 18000, "maize": 12000, "apple": 25000, "banana": 15000,
    "mango": 20000, "grapes": 23000, "watermelon": 14000, "cotton": 16000,
    "coffee": 27000, "jute": 11000, "lentil": 13000, "mungbean": 12500,
    "pigeonpeas": 14500, "kidneybeans": 16500, "chickpea": 15500,
    "blackgram": 11500, "pomegranate": 22000, "muskmelon": 13500,
    "orange": 19000, "papaya": 17500, "coconut": 18500
}

# Model Definition from aicrop.py
class CropModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CropModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Risk Assessment from aicrop.py
def assess_risk(features):
    temp, humidity, ph = features[3], features[4], features[5]
    risk = 0
    if temp > 35: risk += 2
    if humidity > 85: risk += 1
    if ph < 5.5 or ph > 7.5: risk += 2
    score = min(risk, 5)

    if score == 0:
        label = "Very Low"
    elif score <= 2:
        label = "Moderate"
    else:
        label = "High"
    return label

# Profit Estimation from aicrop.py
def estimate_profit(crop_name):
    return profit_table.get(crop_name.lower(), 10000)

# Load the trained model and preprocessing objects
try:
    model_path = os.path.join(os.path.dirname(__file__), "crop_model.pth")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    encoder_path = os.path.join(os.path.dirname(__file__), "encoder.pkl")
    
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    input_dim = 7 # Based on your model's input
    output_dim = len(encoder.classes_)
    
    model = CropModel(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    print("Model and preprocessing objects loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    # Exit or handle the error gracefully
    model, scaler, encoder = None, None, None


# Input schema for FastAPI
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict(data: CropInput):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
    
    input_features = np.array([[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]])
    
    # Preprocess and predict using the model
    features_scaled = scaler.transform(input_features)
    tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        crop = encoder.inverse_transform(pred.numpy())[0]

    # Use the logic from aicrop.py for dynamic output
    risk_factors = assess_risk(input_features[0])
    annual_income = estimate_profit(crop)
    
    return {
        "recommended_crop": crop,
        "risk_factors": risk_factors,
        "annual_income": f"₹{annual_income}"
    }

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🌱"}
