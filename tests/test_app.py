import pytest
import json
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Sign Language Detection" in rv.data

@patch('app.TrainPipeline.run_pipeline')
def test_train_route_success(mock_run_pipeline, client):
    mock_run_pipeline.return_value = None
    rv = client.get('/train')
    assert rv.status_code == 200
    assert b"Training Successful!!" in rv.data

@patch('app.TrainPipeline.run_pipeline')
def test_train_route_failure(mock_run_pipeline, client):
    mock_run_pipeline.side_effect = Exception("Training error")
    rv = client.get('/train')
    assert rv.status_code == 500
    assert b"Training Failed: Unexpected Error" in rv.data

@patch('app.decodeImage')
@patch('os.path.exists')
@patch('os.system')
@patch('app.encodeImageIntoBase64')
def test_predict_route_success(mock_encodeImageIntoBase64, mock_os_system, mock_path_exists, mock_decodeImage, client):
    mock_path_exists.side_effect = [True, True]
    mock_encodeImageIntoBase64.return_value = b"image_data"
    data = json.dumps({"image": "base64string"})
    rv = client.post('/predict', data=data, content_type='application/json')
    assert rv.status_code == 200
    assert b"image_data" in rv.data

def test_predict_route_no_image(client):
    data = json.dumps({})
    rv = client.post('/predict', data=data, content_type='application/json')
    assert rv.status_code == 400
    assert b"Invalid input: No image field in request data" in rv.data

@patch('os.path.exists')
@patch('app.decodeImage')
def test_predict_route_yolo_not_found(mock_decodeImage, mock_path_exists, client):
    mock_path_exists.side_effect = [False]
    data = json.dumps({"image": "base64string"})
    rv = client.post('/predict', data=data, content_type='application/json')
    assert rv.status_code == 500
    assert b"Internal Server Error: YOLOv5 directory not found" in rv.data

@patch('os.system')
@patch('os.path.exists')
def test_predict_route_output_not_found(mock_path_exists, mock_os_system, client):
    mock_path_exists.side_effect = [True, False]
    data = json.dumps({"image": "base64string"})
    rv = client.post('/predict', data=data, content_type='application/json')
    assert rv.status_code == 500
    assert b"Internal Server Error: Detection output not found" in rv.data

@patch('os.system')
def test_predict_live_success(mock_os_system, client):
    rv = client.get('/live')
    assert rv.status_code == 200
    assert b"Camera starting!!" in rv.data
