from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

#AYARLAR 
load_dotenv()

MODEL_PATH = "models/model3_linear_svc.pkl"
VEC_PATH = "models/vectorizer_3.pkl"
DATA_PATH = "datasets/tr_yemekler_arttirilmis_temiz.csv"
FOODNAME_JSON = "datasets/foodname_map.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🔹 Model ve Vectorizer Yükleniyor...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# JSON dosyası
try:
    with open(FOODNAME_JSON, "r", encoding="utf-8") as f:
        foodname_map = json.load(f)
except FileNotFoundError:
    foodname_map = {}

# Veri seti
try:
    DF = pd.read_csv(DATA_PATH)
except:
    DF = None

#FASTAPI
app = FastAPI(title="Akıllı Tarif Asistanı", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GİRDİ MODELİ
class PredictRequest(BaseModel):
    ingredients: str

# TAHMİN ENDPOINTİ 
@app.post("/predict")
def predict(req: PredictRequest):
    text = req.ingredients.strip()
    if not text:
        return {"ok": False, "error": "Malzeme girilmedi."}

    # Vektörleştir ve tahmin yap
    vec = vectorizer.transform([text])
    foodname = model.predict(vec)[0]

    message = foodname_map.get(
        foodname,
        f"Bu malzemelerle '{foodname}' tarifi yapılabilir."
    )

    # GPT'den kısa tarif al
    prompt = f"""
    Sen deneyimli bir Türk aşçısısın.
    Cevabını tamamen Türkçe yaz, İngilizce kelime kullanma.
    Malzemeler: {text}
    Yemek Adı: {foodname}
    Bu malzemelere uygun kısa (4-5 adımlık) bir Türkçe tarif yaz.
    Her adımı yeni satıra yaz.
    """

    try:
        gpt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sen bir Türk aşçısısın ve sadece Türkçe konuşursun."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        recipe_text = gpt_resp.choices[0].message.content.strip()
    except Exception as e:
        print("GPT Hatası:", e)
        recipe_text = "Şu anda tarif önerisi alınamadı."

    # JSON çıktısı (ön tarafa gönderilecek)
    return {
        "ok": True,
        "foodname": foodname,
        "message": message,
        "example": {
            "title": foodname,
            "short_recipe": recipe_text
        }
    }

# Ana sayfa testi
@app.get("/")
def home():
    return {"message": "Akıllı Tarif Asistanı API çalışıyor! POST /predict endpointini kullanabilirsiniz."}

#ÇALIŞTIRMA 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
