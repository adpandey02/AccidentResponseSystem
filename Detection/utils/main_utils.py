import os.path
import sys
import yaml
import base64
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#import sqlite3
import pymongo
from geopy.distance import geodesic
import geocoder
from geopy.geocoders import Nominatim
from twilio.rest import Client
import datetime
import time
import os
import torch
import cv2
from ultralytics import YOLO
from Detection.exception import AppException
import supervision as sv
from Detection.exception import AppException
from Detection.logger import logging




def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e
    



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

        start_time = time.time()
        prediction_interval = 10
        while True:

            ret, frame = cap.read()

            elapsed_time = time.time() - start_time
            if elapsed_time >= prediction_interval:
                results = self.predict(frame)
                frame, class_ids = self.plot_bboxes(results, frame)

            # logic checks to prevent too many emails and notifications spamming 
            
                if len(class_ids) > 0:
                    if not self.email_sent or not self.notification_posted:

                        ### write function to send email and notification here --@Abeer-- , i am writing passing for now ###
                        #result = response_obj.get_incident_details(camera_latitude, camera_longitude, incident_type)

                        self.email_sent = True
                        self.notification_posted = True
                        
                else:
                    self.email_sent = False 
                    self.notification_posted = False          #reset flag when no accident is detected 

                    
                cv2.imshow('accident detection', frame)
                start_time = time.time()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





class Responses:
    incident_mapping = {
        "fire": {
            "collections": ["firestations", "hospitals", "policestations"],
            "service_labels": ["Fire Station", "Hospital", "Police Station"],
        },
        "fight": {
            "collections": ["policestations"],
            "service_labels": ["Police Station"],
        },
        "accident": {
            "collections": ["hospitals", "policestations"],
            "service_labels": ["Hospital", "Police Station"],
        },
    }
    
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["SERVICES"]
        self.camera_latitude, self.camera_longitude = 28.55678, 77.28322
        self.final_services = []

    def get_location(self, lat, lon):
        geoLoc = Nominatim(user_agent="GetLoc")
        location = geoLoc.reverse((lat, lon))
        return location.address

    def calculate_nearest_services(self, incident_type):
        if incident_type in self.incident_mapping:
            incident_data = self.incident_mapping[incident_type]
            service_labels = self.incident_mapping[incident_type]["service_labels"]

            nearest_services = {}
            for i, collection_name in enumerate(incident_data["collections"]):
                nearest_services[collection_name] = []

                for service in self.db[collection_name].find():
                    service_location = (service["LATITUDE"], service["LONGITUDE"])
                    distance = geodesic((self.camera_latitude, self.camera_longitude), service_location).kilometers

                    nearest_services[collection_name].append({
                        "name": service["NAME"],
                        "service": incident_data["service_labels"][i],
                        "distance": distance,
                    })

            for collection_name in nearest_services:
                nearest_services[collection_name].sort(key=lambda x: x["distance"])

            for collection_name in nearest_services:
                if nearest_services[collection_name]:
                    nearest_service = nearest_services[collection_name][0]
                    self.final_services.append(nearest_service)

        else:
            print(f"Invalid incident type: {incident_type}")


    def get_service_location(self):
        latitude = []
        longitude = []
        for service in self.final_services:
            service_name = service['name']
            service_type = service['service']

            if service_type == "Fire Station":
                collection_name = "firestations"
            elif service_type == "Hospital":
                collection_name = "hospitals"
            elif service_type == "Police Station":
                collection_name = "policestations"

            service_data = self.db[collection_name].find_one({"NAME": service_name})

            if service_data:
                latitude_number = service_data.get('LATITUDE', '')
                longitude_number = service_data.get('LONGITUDE', '')

                latitude.append(latitude_number)
                longitude.append(longitude_number)

        return latitude, longitude

    def generate_directions_urls(self, latitude, longitude):
        directions_urls = []

        for i in range(len(latitude)):
            start_latitude = latitude[i]
            start_longitude = longitude[i]

            url = f"https://www.google.com/maps/dir/{start_latitude},{start_longitude}/{self.camera_latitude},{self.camera_longitude}"
 
            directions_urls.append(url)

        return directions_urls

    def send_sms(self, to_phone_number, message):
        account_sid = 'ACa7f320a488f1812d0b10de2be6be2217'
        auth_token = 'cba2eaf51227cb598e0a3aea479a5b86'
        from_phone_number = '+18142994366'

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_phone_number,
            to=to_phone_number
        )
     

    def notify_via_sms(self, incident_type, incident_location, current_datetime, service_name, phone_number, url):
        message = f'''
        Dear {service_name},

        An incident has been detected at the following location:

        Type of Incident: {incident_type}
        Latitude: {self.camera_latitude}
        Longitude: {self.camera_longitude}
        Address of Accident: {incident_location}
        Google Map Link (Route): {url}

        Please respond promptly to this location.

        Additional Information:
        Date and Time: {current_datetime}

        Sincerely,
        Your Emergency Notification System
        '''

#         self.send_sms(phone_number, message)
        print(f"SMS sent to {service_name}")

    def send_email(self, subject, message, recipient_email):
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        sender_email = 'aznakhkydseu@gmail.com'
        sender_password = 'azal otlv kvkl uzjp'

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print('Email sent successfully.')
            server.quit()
        except Exception as e:
            print('Email sending failed.')
            print(str(e))
     
    def notify_via_email(self, incident_type, accident_location, current_datetime, service_name, email_address, url):
        subject = 'Emergency Notification'

        message = f'''
        Dear {service_name},

        An incident has been detected at the following location:

        Type of Incident: {incident_type}
        Latitude: {self.camera_latitude}
        Longitude: {self.camera_longitude}
        Address of Incident: {accident_location}

        Google Map Link (Route): {url}

        Please respond promptly to this location.

        Additional Information:
        Date and Time: {current_datetime}

        Sincerely,
        Your Emergency Notification System
        '''

#         self.send_email(subject, message, email_address)
        print(f"Email sent to {service_name}")

    def get_collection_name(self, service_type):
        if service_type == "Fire Station":
            return "firestations"
        elif service_type == "Hospital":
            return "hospitals"
        elif service_type == "Police Station":
            return "policestations"
        else:
            raise ValueError(f"Invalid service type: {service_type}")

    def get_service_data(self, service):
        return self.db[self.get_collection_name(service['service'])].find_one({"NAME": service['name']})
    
    def get_incident_details(self, camera_latitude, camera_longitude, incident_type):
        self.final_services = []  # Reset final_services for each query

        # Calculate nearest services based on the provided incident type
        self.calculate_nearest_services(incident_type)

        # Get incident location details
        incident_location = self.get_location(camera_latitude, camera_longitude)

        # Generate URLs for directions
        latitude, longitude = self.get_service_location()
        directions_urls = self.generate_directions_urls(latitude, longitude)

        confirmation_lines_email = []
        confirmation_lines_sms = []
        current_datetime = datetime.datetime.now()

        result = []

        for service, url in zip(self.final_services, directions_urls):
            service_name = service['name']
            service_data = self.get_service_data(service)
            phone_number = service_data.get('PHN NO.', '')
            email_address = service_data.get('EMAIL ID', '')

            urls = f" {url} "
            confirmation_email = f"Email sent to {service_name}"
            confirmation_sms = f"SMS sent to {service_name}"
            email_address = f" {email_address} "
            service_details = {
                'Name': service_name,
                'Service Type': service['service'],
                'Distance': service['distance'],
                'Phone Number': phone_number,
                'Email Address': email_address,
                'Direction URL': urls,
                'Email confirmation': confirmation_email,
                'SMS confirmation': confirmation_sms
            }

            result.append(service_details)

            
            self.notify_via_email(incident_type, incident_location, current_datetime, service_name, email_address, urls)
            self.notify_via_sms(incident_type, incident_location, current_datetime, service_name, phone_number, urls)

        return {
            'TYPE OF INCIDENT': incident_type,
            'DATE and TIME OF INCIDENT': current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'LATITUDE OF INCIDENT': camera_latitude,
            'LONGITUDE OF INCIDENT': camera_longitude,
            'ADDRESS OF INCIDENT': incident_location,
            'SERVICE DETAILS': result
        }





def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise AppException(e, sys)
    




def correct_paths_in_yaml(yaml_file_path):
    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update parameters
    config['test'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\test\\images'
    config['train'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\train\\images'
    config['val'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\valid\\images'

    # Write the updated content back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(config, file)



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

    
    