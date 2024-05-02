import os
import sys
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from PIL import Image
import numpy as np
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from rs_mamba_ss import RSM_SS
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

def train_net(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    test_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/test/image/',
                                labels_dir=f'{ph.root_dir}/{dataset_name}/test/label/',
                                train=False)
    # 2. Create data loaders

    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')
    net = RSM_SS(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
                 ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    logging.info(f'Model loaded from {ph.load}')
    print(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')

    # 在循环开始前创建保存图片的文件夹
    os.makedirs('output_images', exist_ok=True)

    with torch.no_grad():
        for batch_img1, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            labels = labels.float().to(device)

            b, c, h, w = batch_img1.shape

            # labels的形状应为B, H, W, 而输入labels多余了一个维度(通道数 3),在第四维,因此利用narrow 与 squeeze删除
            labels = labels.narrow(3, 1, 1)
            labels = labels.squeeze(3)

            # ss_preds = net(batch_img1, log=True, img_name=name)
            ss_preds = net(batch_img1)
            ss_preds = torch.sigmoid(ss_preds)

            # Calculate and log other batch metrics
            ss_preds = ss_preds.float()
            labels = labels.int().unsqueeze(1)

            metric_collection.update(ss_preds, labels)

            # 保存输出图像
            for i in range(ss_preds.shape[0]):
                pred_log = torch.round(ss_preds[i]).cpu().clone().float()
                result_pil_image = transforms.ToPILImage()(pred_log)
                result_pil_image.save(f'output_images/{name[i]}.png')  # 保存图像，name[i] 为图像的文件名

            # clear batch variables from memory
            del batch_img1, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        train_net(dataset_name='palsar')
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
