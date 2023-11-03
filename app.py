import sys,os
import torch
import cv2
from time import time
from ultralytics import YOLO
from Detection.exception import AppException
import supervision as sv
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from Detection.constant.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"






@app.route("/")
def home():
    return render_template("index.html")



class AccidentDetector:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.notification_posted = False
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        print(f'using device : {self.device}')
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=3, text_thickness=3)

    
    def load_model(self):
        model = YOLO('D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\model_trainer\\best.pt')
        model.fuse()
        return model
    

    def predict(self, frame):
        results = self.model(frame)
        return results
    

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # extract detection for accidents
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(class_id)


        # setup detections for visualization

        detections = sv.Detections.from_ultralytics(results[0])
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        return frame, class_ids
    

    def begin(self):

        cap = cv2.VideoCapture(self.capture_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        while True:

            ret, frame = cap.read()
            results = self.predict(frame)
            frame, class_ids = self.plot_bboxes(results, frame)

            # logic checks to prevent too many emails and notifications spamming 
            if len(class_ids) > 0:
                if not self.email_sent or not self.notification_posted:

                    ### write function to send email and notification here --@Abeer-- , i am writing passing for now ###
                    
                    self.email_sent = True
                    self.notification_posted = True
            else:
                self.email_sent = False 
                self.notification_posted = False          #reset flag when no accident is detected 

            cv2.imshow('accident detection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



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

