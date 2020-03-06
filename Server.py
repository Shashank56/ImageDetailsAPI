from imageai.Detection import ObjectDetection
from flask import Flask, jsonify,request,render_template
from flask_cors import CORS, cross_origin  
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)
detector = ObjectDetection()
execution_path = os.getcwd()
CORS(app, support_credentials=True)
@cross_origin(supports_credentials=True)

def Detection(img):
    print(img)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path ,img), output_image_path=os.path.join(execution_path , "2_detected.jpg"), minimum_percentage_probability=40)
    animals = ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
    Data = {"animals":0,"HumanFaces":0,"objects":0}
    for eachObject in detections:
        if eachObject["name"] in animals:
            Data["animals"] +=1
            # print("animal" , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        elif(eachObject["name"]=="person"):
            Data["HumanFaces"] +=1
            # print("person" , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        else:
            Data["objects"] +=1
            # print("object" , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print(Data)
    return jsonify(Data)

@app.route('/')
def main_route():
    return '<form action="getImageDetails" method="post" enctype="multipart/form-data">Select image to upload:<input type="file" name="file" id="fileToUpload"><br/><input type="submit" value="Upload Image" name="submit"></form>'

@app.route('/getImageDetails', methods=['POST'])
def getImageDetails():
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(filename))
        out = Detection(filename)
        return out
   
if __name__ == '__main__':
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    app.run(host='0.0.0.0',port='5000',debug=True)