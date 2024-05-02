import io
import flask
from flask_cors import CORS
import torch
from rs_mamba_ss import RSM_SS
from torchvision import transforms
from PIL import Image
from utils.path_hyperparameter import ph
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import base64



app = flask.Flask(__name__)
CORS(app)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = None

def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global net
    net = RSM_SS(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
                 ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,
                 patchembed_version=ph.patchembed_version)
    net = net.to(device)
    pth_path = r'/root/hx/RSmamba/Official_Remote_Sensing_Mamba-main/semantic_segmentation_mamba/train_palsar_256_0.001_checkpoint/checkpoint_spiral_epoch89.pth'
    net.load_state_dict(torch.load(pth_path, map_location=device)['net'])
    net.eval()

def prepare_image(img,target_size):
    """Do image preprocessing before prediction on any data.

    :param img:       original image
    :return:
                        preprocessed image tensor
    """
    # 加载图像
    img = np.array(img).astype(np.uint8)

    # 定义一个转换操作，将图像转换为 PyTorch 张量
    normalize = A.Compose([
        A.Normalize()
    ])

    to_tensor = ToTensorV2()

    # 应用转换操作并获取图像张量
    img = normalize(image=img)['image']
    img_tensor = to_tensor(image=img)['image'].contiguous()
    # 添加batch_size维度
    img_tensor = img_tensor.reshape(1, 3, target_size, target_size)

    # print(img_tensor.shape)
    #将img_tensor放在device
    img_tensor = img_tensor.to(device)

    return img_tensor

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if 'imageFile' in flask.request.files:
            # Decode the base64 string to image bytes
            image_file = flask.request.files['imageFile']
            image = Image.open(image_file)

            # Convert image bytes to PIL format
            print(image.size)

            # Preprocess the image and prepare it for classification.
            image_tensor = prepare_image(image, target_size=256)

            # Classify the input image and then initialize the list of predictions to return to the client.
            with torch.no_grad():
                pred = net(image_tensor)
                pred = torch.sigmoid(pred)
                pred_log = torch.round(pred[0]).cpu().clone().float()

                # 将 PyTorch 张量转换为 PIL 图像
                result_pil_image = transforms.ToPILImage()(pred_log)

                # 将 PIL 图像转换为字节流
                image_bytes = io.BytesIO()
                result_pil_image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()

                # 使用 base64 编码将字节数据转换为字符串
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')

                # 将图片的字节流添加到返回的数据字典中
                data["prediction_image"] = encoded_image

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

load_model()
app.run(host='172.26.94.21',port='5001')
#服务器ip
PyTorch_REST_API_URL = 'http://172.26.94.21:5001/predict'