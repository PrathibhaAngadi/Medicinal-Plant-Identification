from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8507/v1/models/medicinal_plant_model:predict"

CLASS_NAMES = ["Alpinia Galanga (Rasna)","Amaranthus Viridis (Arive-Dantu)","Artocarpus Heterophyllus (Jackfruit)","Azadirachta Indica (Neem)","Basella Alba (Basale)","Brassica Juncea (Indian Mustard)","Carissa Carandas (Karanda)","Citrus Limon (Lemon)","Ficus Auriculata (Roxburgh fig)","Ficus Religiosa (Peepal Tree)","Hibiscus Rosa-sinensis","Jasminum (Jasmine)","Mangifera Indica (Mango)","Mentha (Mint)","Moringa Oleifera (Drumstick)","Muntingia Calabura (Jamaica Cherry-Gasagase)","Murraya Koenigii (Curry)","Nerium Oleander (Oleander)","Nyctanthes Arbor-tristis (Parijata)","Ocimum Tenuiflorum (Tulsi)","Piper Betle (Betel)","Plectranthus Amboinicus (Mexican Mint)","Pongamia Pinnata (Indian Beech)","Psidium Guajava (Guava)","Punica Granatum (Pomegranate)","Syzygium Cumini (Jamun)","Santalum Album (Sandalwood)","Syzygium Jambos (Rose Apple)","Tabernaemontana Divaricata (Crape Jasmine)","Trigonella Foenum-graecum (Fenugreek)"]

@app.get("/ping")
async def ping():
    return "I'm working!!!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    json_data = {
        "instances": img_batch.tolist()
    }
    
    response = requests.post(endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return{
        "class": predicted_class,
        "confidence": (confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)