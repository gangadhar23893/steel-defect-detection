from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
from datetime import datetime

from inference import load_model, predict
from s3_utils import download_file,list_files

app = FastAPI()

model = load_model("models/corrosion_model.pth")

BUCKET_NAME = "steel-defect-detection-gangadhar"


class S3Request(BaseModel):
    s3_key: str


@app.post("/predict")
def predict_from_s3(request: S3Request):
    local_path = f"temp/{request.s3_key.split('/')[-1]}"

    os.makedirs("temp", exist_ok=True)

    # Download from S3
    download_file(BUCKET_NAME, request.s3_key, local_path)

    # Run inference
    result = predict(local_path, model)

    return {"prediction": result}

@app.post('/batch_predict')
def batch_predict():
    s3_files = list_files(bucket_name=BUCKET_NAME,prefix="input_images")
    results=[]
    alerts=[]
    for s3_key in s3_files:
        if s3_key.endswith('/'):
            continue
        filename = s3_key.split('/')[-1]
        local_path = f"temp/{filename}"
        print(f"Downloading:{s3_key}")
        download_file(BUCKET_NAME,s3_key,local_path)
        print(f"Running inference : {filename}")
        prediction = predict(local_path,model)
        result_entry = {
            "image":filename,
            "prediction":prediction,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result_entry)
        if prediction == "Defect":
            alert_entry = {
                "image":filename,
                "alert":"Corossion_detected",
                "severity":"High",
                "timestamp": datetime.now().isoformat()
            }
            alerts.append(alert_entry)
    #save results to json
    with open("results/results.json","w") as f:
        json.dump(results,f,indent=4)

    #save results to json
    with open("results/alerts.json","w") as f:
        json.dump(alerts,f,indent=4)

    return{
        "total_images":len(results),
        "total_laerts" : len(alerts),
        "message":"Batch Inference Completed"
    }
