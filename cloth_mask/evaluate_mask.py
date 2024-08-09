import torch.nn as nn
import os
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from cloth_mask.dataset_mask import VITONDataset
from cloth_mask.utils_mask import test_transform
from cloth_mask.model_mask import UNet
import numpy as np


def get_arguments():
    '''
        Cloth-mask provied from dekiru
        return:
            A list of parsed arguments
    '''

    parser = argparse.ArgumentParser(description="Mask cloth")
    parser.add_argument('--model-path', type=str, default='pre_trained/model_ep_30.pth')
    parser.add_argument('--input', type=str, default='Database/val/cloth', help='path of input image cloth')
    parser.add_argument('--output', type=str, default='Database/val/cloth-mask', help='path of output mask image')
    return parser.parse_args()


def get_palette(num_cls = 2):
    palette = [0, 0, 0, 255, 255, 255]  # Black and White
    return palette

def get():
    args = get_arguments()

    model = UNet(1)
    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    dataset = VITONDataset(root_dir=args.input, transform=test_transform)
    dataloader = DataLoader(dataset)
    device = torch.device('cpu')

    palette = get_palette(2)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image, img_name = batch
            img_name = img_name[0]
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            x = image.to(device).float()
            y_hat = model(x).squeeze(0) #(1, 1, H, W) -> (H, W)
            y_hat_mask = y_hat.sigmoid().round().long().cpu().numpy()
            output_img = Image.fromarray(y_hat_mask[0].astype(np.uint8))
            # mask_result_path = os.path.join(args.output, img_name[:-4] + '.png')
            output_img.putpalette(palette)
            # output_img.save(mask_result_path)
            output_img_rgb = output_img.convert('L')
            jpeg_result_path = os.path.join(args.output, img_name[:-4] + '.jpg')
            output_img_rgb.save(jpeg_result_path, format='JPEG')
    return

def execute_mask():
    get()
