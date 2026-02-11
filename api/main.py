from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Track&Care - Localização Indoor")

# Caminhos dos artefatos (ajustados para rodar dentro da pasta api ou via Docker)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_final.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'model_columns.pkl')

# Carregar artefatos globalmente para performance
try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")

class TelemetryInput(BaseModel):
    imei: str
    sensor_latlong: str
    sensor_room: str
    rssi: float

@app.get("/")
def health_check():
    return {"status": "online", "message": "Modelo Track&Care pronto para inferência"}

@app.post("/predict")
def predict(data: TelemetryInput):
    try:
        # 1. Processamento das coordenadas
        lat, lon = map(float, data.sensor_latlong.split(','))
        
        # 2. Clipping do RSSI (mesmo tratamento do treino)
        rssi_clean = max(min(data.rssi, -30), -100)
        
        # 3. Criação do DataFrame e One-Hot Encoding
        input_dict = {
            'rssi_clean': rssi_clean,
            'sensor_lat': lat,
            'sensor_long': lon,
            'sensor_room': data.sensor_room,
            'imei': data.imei
        }
        input_df = pd.DataFrame([input_dict])
        
        # Aplicar Get Dummies
        input_df = pd.get_dummies(input_df)
        
        # Reindexar para garantir que todas as colunas do treino existam (preenchendo com 0)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # 4. Predição
        pred_idx = model.predict(input_df)[0]
        # Cálculo de confiança (probabilidade da classe escolhida)
        probs = model.predict_proba(input_df)[0]
        confidence = np.max(probs)
        
        room_name = le.inverse_transform([pred_idx])[0]
        
        return {
            "predicted_room": room_name,
            "confidence": round(float(confidence), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")