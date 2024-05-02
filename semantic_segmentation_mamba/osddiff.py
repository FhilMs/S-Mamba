import torch
from rs_mamba_ss import RSM_SS
from torchvision import transforms
from PIL import Image
from utils.path_hyperparameter import ph
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


if __name__ == '__main__':
    # 获取命令行参数
    image_path = r"/root/hx/sar.png"
    result_image_path = r"/root/hx/pred.png"

    # 加载图像
    img = Image.open(image_path)
    img = np.array(img).astype(np.uint8)
    # 将图像转换为灰度模式
    # img = img.convert('L')

    # 定义一个转换操作，将图像转换为 PyTorch 张量
    normalize = A.Compose([
        A.Normalize()
    ])

    to_tensor = ToTensorV2()

    img = normalize(image=img)['image']
    img_tensor = to_tensor(image=img)['image'].contiguous()
    # 应用转换操作并获取图像张量

    img_tensor = img_tensor.reshape(1,3,256,256)
    print(img_tensor.shape)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)

    net = RSM_SS(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
                 ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)
    net = net.to(device)
    p = r'/root/hx/RSmamba/Official_Remote_Sensing_Mamba-main/semantic_segmentation_mamba/train_palsar_256_0.001_checkpoint/checkpoint_spiral_epoch99.pth'
    net.load_state_dict(torch.load(p, map_location=device)['net'])

    with torch.no_grad():
        net.eval()
        pred = net(img_tensor)
        pred = torch.sigmoid(pred)
        pred_log = torch.round(pred[0]).cpu().clone().float()

        # 将 PyTorch 张量转换为 PIL 图像
        result_pil_image = transforms.ToPILImage()(pred_log)
        # 保存图像到指定路径
        result_pil_image.save(result_image_path)
    print("over")