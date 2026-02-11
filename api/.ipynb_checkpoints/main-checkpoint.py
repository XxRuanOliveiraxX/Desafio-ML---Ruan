from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Track&Care - Localização Indoor")

# Carregar artefatos
model = joblib.load('../models/model_v5.pkl')
le = joblib.load('../models/label_encoder.pkl')

class TelemetryInput(BaseModel):
    imei: str
    sensor_latlong: str
    sensor_room: str
    rssi: float

@app.get("/")
def read_root():
    return {"status": "online", "model": "Random Forest V5"}

@app.post("/predict")
def predict(data: TelemetryInput):
    try:
        # 1. Pré-processamento (Igual ao do Treino)
        # Separar Lat e Long
        lat, lon = map(float, data.sensor_latlong.split(','))
        
        # Clipping do RSSI
        rssi_clean = max(min(data.rssi, -30), -100)
        
        # Criar DataFrame para o Pipeline (deve ter as colunas originais)
        input_df = pd.DataFrame([{
            'rssi_clean': rssi_clean,
            'sensor_lat': lat,
            'sensor_long': lon,
            'sensor_room': data.sensor_room,
            'imei': data.imei
        }])
        
        # 2. Predição
        pred_idx = model.predict(input_df)[0]
        confidence = np.max(model.predict_proba(input_df))
        
        room_name = le.inverse_transform([pred_idx])[0]
        
        return {
            "predicted_room": room_name,
            "confidence": round(float(confidence), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))