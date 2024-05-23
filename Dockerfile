FROM python:3.7-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
# Explicitly copy the yolov5 directory to /app/yolov5 in the container
COPY yolov5 /app/yolov5

# Debugging step to print the directory structure
RUN ls -l /app && ls -l /app/yolov5

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]