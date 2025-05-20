"""
    -----------------------------------------------------------
    ECE441 - Illinois Institute of Technology
    Author: Alae Moudni
    Email: amoudni@hawk.iit.edu
    Date: October 2023
    
    Description:
    This code is a part of the ECE441 project. Unauthorized copying,
    distribution, or any form of plagiarism without explicit permission is
    strictly prohibited. All rights reserved.
    -----------------------------------------------------------
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import logging
import random
from src.architecture import YOLOModel
import asyncio
from starlette.concurrency import run_in_threadpool
import threading
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize models and warmup
yolo_model = YOLOModel()

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.clients = []
        self.frame = None
        thread = threading.Thread(target=self.get_frame, args=())
        thread.daemon = True
        thread.start()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if not success:
                break
            if image is None:
                print("No image")
                break

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            #image = yolo_model.detect_and_track_objects2(image)
            #image = yolo_model.detect_objects(image)
            #image = yolo_model.detect_pose(image)
            #image = yolo_model.detect_objects_and_draw(image)
            #result = loop.run_until_complete(yolo_model.pipeline(image))
            #result, image = loop.run_until_complete(yolo_model.pipeline_and_draw(image))
            # Uncomment the line below to use pothole detection instead
            result, image = loop.run_until_complete(yolo_model.pipeline_pothole_and_draw(image))
            print(result)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            self.frame = (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            self.broadcast()

    def broadcast(self):
        for q in self.clients:
            q.put(self.frame)

    def get_stream(self):
        q = Queue(maxsize=1)
        self.clients.append(q)
        try:
            while True:
                yield q.get()
        finally:
            self.clients.remove(q)

camera = VideoCamera()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(camera.get_stream(), media_type="multipart/x-mixed-replace;boundary=frame")