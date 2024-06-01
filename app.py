import flask

import cv2 
import numpy as np 
from flask import Flask, render_template, request, redirect
from werkzeug.utils import send_from_directory
import os 

import pandas as pd

import cvzone

from ultralytics import YOLO


folder_path = 'runs/detect'
subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
latest_subfolder = max(subfolders, key= lambda x: os.path.getctime(os.path.join(folder_path, x)))
directoryf = folder_path+'/'+latest_subfolder
# , redirect, render_template
app = Flask(__name__)

detect_folder = 'runs\detect\predict'

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/enumeration')
def enumeration():
    return render_template("enumeration.html")

@app.route('/classification')
def classification():
    return render_template("classification.html")

@app.route('/weather')
def weather():
    return render_template("weather.html")

@app.route('/sumitagale')
def sumitagale():
    return redirect('http://sumitagale.netlify.app')

@app.route('/devidasdukale')
def devidasdukale():
    return redirect('https://devidasdukale.000webhostapp.com/')



@app.route('/enumeration', methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file'] # reterivers uploaded file object from the request. 
            basepath = os.path.dirname(__file__) # get directory path of current script
            filepath = os.path.join(basepath,'uploads',f.filename) 
            print("upload folder is ", filepath)

            f.save(filepath)

            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.',1)[1].lower()
            allowed_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']

            if file_extension in allowed_extensions:
                img = cv2.imread(filepath)


                yolo = YOLO('yolov8_model.pt')
                detections = yolo.predict(source=img, conf=0.25, save=True)

                # a = detections[0].boxes.data
                folder_path = 'runs/detect'
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                latest_subfolder = max(subfolders, key= lambda x: os.path.getctime(os.path.join(folder_path, x)))
                directoryf = folder_path+'/'+latest_subfolder

                

                tree_count = 0

                a = detections[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                object_classes = []

                for index, row in px.iterrows():
                    x1=int(row[0])  # Retriving bounding box co-ordinates 
                    y1=int(row[1])
                    x2=int(row[2])
                    y2=int(row[3])
                    d=int(row[5])       # extracts the class index (d) for the detected object from the DataFrame row.
                    c = class_list[d]  
                    obj_class = class_list[d]
                    object_classes.append(obj_class)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(img, f'{obj_class}', (x2, y2), 1, 1)


                print(object_classes)
                print('******************************************************************************************', len(object_classes))

                tree_count = len(object_classes)

                print("printing directory: ",directoryf)
                files = os.listdir(directoryf)
                latest_file = files[0]

                print(latest_file)

                filename = os.path.join(folder_path, latest_subfolder, latest_file)
                print(filename)

                result_image_path = 'image0.jpg'
                return render_template('enumeration.html', result_image_path=result_image_path, tree_count=tree_count)
            else:
                print("Unsupported file format. Please upload a file with valid extensions..")

@app.route('/<path:filename>')
def get_result_image(filename):
    environ = request.environ
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key= lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directoryf = folder_path+'/'+latest_subfolder
    return send_from_directory(directoryf, filename,environ)

app.run(debug=True)