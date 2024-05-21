import logging
from logging.handlers import RotatingFileHandler
import sys, os
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.exception import SignException
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

# Configure logging
log_handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
log_handler.setFormatter(log_formatter)

app = Flask(__name__)
app.logger.addHandler(log_handler)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        app.logger.info("Training completed successfully")
        return "Training Successful!!"
    except Exception as e:
        app.logger.error("Error during training: %s", str(e))
        return "Training Failed"

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")
        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf yolov5/runs")
        app.logger.info("Prediction successful")
    except ValueError as val:
        app.logger.error("Value error: %s", str(val))
        return Response("Value not found inside json data")
    except KeyError as e:
        app.logger.error("Key error: %s", str(e))
        return Response("Key value error incorrect key passed")
    except Exception as e:
        app.logger.error("Unexpected error: %s", str(e))
        result = "Invalid input"
    return jsonify(result)

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        app.logger.info("Camera starting!!")
        return "Camera starting!!"
    except ValueError as val:
        app.logger.error("Value error: %s", str(val))
        return Response("Value not found inside json data")

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
