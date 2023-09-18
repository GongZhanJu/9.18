import base64
import pathlib
import requests
import json
import numpy as np
import math

TF_SERVING_BASE_URL = "YOUR_TF_SERVING_URL"  # Update this to your TensorFlow Serving URL
model_version = "YOUR_MODEL_VERSION"  # Update this if you're using a specific model version

def predict_image(images):
    bimages = []
    for image in images:
        with open(image, 'rb') as fimage:
            content = fimage.read()
        bimage = base64.urlsafe_b64encode(content).decode()
        bimages.append(bimage)
    req_data ={
      'inputs': bimages,
    }
    response = requests.post(TF_SERVING_BASE_URL + f'v1/models/slot1/versions/{model_version}:predict',
                             json=req_data,
                             headers={"content-type": "application/json"})
    if response.status_code != 200:
        raise RuntimeError('Request tf-serving failed: ' + response.text)
    resp_data = json.loads(response.text)
    if 'outputs' not in resp_data or type(resp_data['outputs']) is not list:
        raise ValueError('Malformed tf-serving response')
    outputs = np.argmax(resp_data['outputs'], axis=1).tolist()
    return outputs

def test_image_model(base_dir, dir_name, code, batch_size=10):
    images = list(pathlib.Path(base_dir).joinpath(dir_name, str(f'Category{code}')).glob('./*.png'))
    codes = []
    for step in range(math.ceil(len(images)/batch_size)):
        outputs = predict_image(images[step*batch_size:(step+1)*batch_size])
        for i, o in zip(images, outputs):
            if o != code:
                print(f'Error picture in {dir_name} category {code}:', i)
        codes.extend(outputs)
    accuracy = round(codes.count(code) / len(codes), 4)
    return accuracy, codes

base_directory = "./images"
dirs_to_test = ['train', 'val', 'test']

for dir_name in dirs_to_test:
    for category in [0, 1]:
        accuracy, codes = test_image_model(base_directory, dir_name, category)
        print(f'Accuracy rate of {dir_name} category {category}', accuracy)
        print(f'Test results of {dir_name} category {category}', codes)
