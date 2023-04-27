from fastapi import File, UploadFile
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from fastapi.responses import FileResponse
from datetime import datetime
import shutil
import os
import json
from fastapi.responses import JSONResponse
from json import dumps

app = FastAPI()

#-------------------------Read and save the File--------------------------------
@app.post("/upload")
def upload(file: UploadFile = File(...)):
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('uploads/', str(ts)+file.filename)
        outputpath = os.path.join('outputs/', os.path.basename(imgpath))
        with open(imgpath, "wb") as buffer:
              shutil.copyfileobj(file.file, buffer)

#--------------------------- Load Model ------------------------------------------
        
        model=YOLO('models/best.pt')

#--------------- Predict and return results in .json format -----------------------

        result=model.predict(imgpath)
        response_ = result[0].tojson(normalize=False)
        response_ = response_.replace("\n", "")        # Preprocess the Returned Results
        predictions = json.loads(response_)
        predictions.save_txt('outputtext/'+os.path.basename(imgpath), save_conf=False)

        return predictions