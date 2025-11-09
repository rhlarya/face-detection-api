from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
 
app = FastAPI(title="Face Detection API", description="Detect faces in uploaded images using OpenCV")
 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
 
@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    image_data = await file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    results = []
    for (x, y, w, h) in faces:
        results.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
 
    return JSONResponse(content={"num_faces": len(faces), "faces": results})

@app.post("/detect_faces/image/")
async def detect_faces_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
 
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    _, buffer = cv2.imencode('.jpg', img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")