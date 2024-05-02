import os

import requests
from PIL import Image
from io import BytesIO
import base64

PyTorch_REST_API_URL = 'http://172.26.94.21:5001/predict'


def predict_result(image_path, save_dir):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': base64.b64encode(image).decode('utf-8')}  # Encode image as base64 string


    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, json=payload).json()
    print(r)
    # Ensure the request was successful.
    if r['success']:
        prediction_image_bytes = base64.b64decode(r['prediction_image'])

        # Convert bytes to PIL image
        prediction_image = Image.open(BytesIO(prediction_image_bytes))
        image_name = os.path.basename(image_path)
        prediction_image_path = os.path.join(save_dir, f'{os.path.splitext(image_name)[0]}_prediction.png')
        prediction_image.save(prediction_image_path)
        print(f'Prediction image saved at: {prediction_image_path}')
        # Display or save the image
        prediction_image.show()

    else:
        print('Request failed')



# Example usage
predict_result('/root/hx/sar.png','/root/hx')
