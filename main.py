import base64
import io
import json
import requests
import cv2
import numpy as np
import PIL.Image as Image
from easyocr import Reader
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

app = FastAPI()




@app.post("/store/")
async def store_data(new_data: dict):
    global data_byte
    data_byte = new_data.get("data")
    return {"messsage": "Data stored successfully"}




@app.get("/license_plate")
async def read_license_plate() -> dict:
    if data_byte is None:
        return {"error": "Image data not found"}

    b = base64.b64decode(data_byte)
    img = Image.open(io.BytesIO(b))
    image = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            n_plate_cnt = approx
            break
    (x, y, w, h) = cv2.boundingRect(n_plate_cnt)
    license_plate = gray[y:y + h, x:x + w]

    reader = Reader(['en'])
    detection = reader.readtext(license_plate)

    if len(detection) == 0:
        text = "Impossible to read the text from the license plate"
    else:
        text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"

    cv2.drawContours(image, [n_plate_cnt], -1, (0, 255, 0), 3)
    cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Save the output image to a file
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)

    result = {"text": text, "output_path": output_path}

    return jsonable_encoder(result)
