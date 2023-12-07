import sys,os
# import torch
# import cv2
# from time import time
# from ultralytics import YOLO
from Detection.exception import AppException
# import supervision as sv
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from Detection.constant.application import APP_HOST, APP_PORT
from Detection.utils.main_utils import Responses
from Detection.utils.main_utils import AccidentDetector

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


# responsse parameters
# response_obj = Responses()
# camera_latitude = 28.56234651631763
# camera_longitude = 77.28039429363223
# incident_type = "accident"


@app.route("/")
def home():
    return render_template("index.html")


# camera opening route
@app.route("/live", methods=['GET'])
#@cross_origin()
def detectLive():
    try:
        detector = AccidentDetector(capture_index=0)
        detector.begin()
        return ('session complete')
    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except Exception as e:
            raise AppException(e, sys)
    



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)

