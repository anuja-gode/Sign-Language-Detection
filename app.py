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

clApp = ClientApp()

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
    except SignException as e:
        app.logger.error("Error during training: %s", str(e))
        return Response(f"Training Failed: {str(e)}", status=500)
    except Exception as e:
        app.logger.error("Unexpected error during training: %s", str(e))
        return Response("Training Failed: Unexpected Error", status=500)

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        if 'image' not in request.json:
            app.logger.error("No image field in request data")
            return Response("Invalid input: No image field in request data", status=400)

        image = request.json['image']
        decodeImage(image, clApp.filename)

        # Ensure YOLOv5 directory exists and is accessible
        if not os.path.exists("yolov5/"):
            app.logger.error("YOLOv5 directory not found")
            return Response("Internal Server Error: YOLOv5 directory not found", status=500)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")
        output_image_path = "yolov5/runs/detect/exp/inputImage.jpg"

        if not os.path.exists(output_image_path):
            app.logger.error("Output image not found after detection")
            return Response("Internal Server Error: Detection output not found", status=500)

        opencodedbase64 = encodeImageIntoBase64(output_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf yolov5/runs")
        app.logger.info("Prediction successful")
        return jsonify(result)
    except ValueError as val:
        app.logger.error("Value error: %s", str(val))
        return Response("Value not found inside json data", status=400)
    except KeyError as e:
        app.logger.error("Key error: %s", str(e))
        return Response("Key value error: incorrect key passed", status=400)
    except Exception as e:
        app.logger.error("Unexpected error: %s", str(e))
        return Response("Invalid input: Unexpected Error", status=500)

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
        return Response("Value not found inside json data", status=400)
    except Exception as e:
        app.logger.error("Unexpected error: %s", str(e))
        return Response("Unexpected Error", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
