from fastapi import FastAPI,File, UploadFile
import shutil
import os
from inference import load_model,predict

app = FastAPI()
model = load_model("models/corrosion_model.pth")

@app.get('/')
def home():
    return { "message":"Corrosion of steel detection"}

@app.post('/predict')
def predict_image(file: UploadFile = File(...)):
    
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    result = predict(file_path,model)
    return {"prediction": result}
