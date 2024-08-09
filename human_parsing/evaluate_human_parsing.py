import os
import torch
import argparse
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from human_parsing.models.pspnet import PSPNet
from human_parsing.dataset.datasets import HumanDataset

import torch
from torchvision import transforms



dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
}

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")
    parser.add_argument('--image_path', type=str, help='Path to image')
    parser.add_argument('--models-path', type=str, default='pre_trained/PSPNet_last', help='Path for storing model snapshots')
    parser.add_argument("--input", type=str, default='./Database/val/person/', help="path of input image folder.")
    parser.add_argument("--output", type=str, default='./Database/val/person-parse', help="path of output image folder.")
    parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
    parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def get():
    args = get_arguments()

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']

    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
    model = nn.DataParallel(model)
    state_dict = torch.load(args.models_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        # transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = HumanDataset(root=args.input, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    palette = get_palette(num_classes)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image, img_name = batch
            img_name = img_name[0]
            # Kiểm tra kích thước của image
            if image.dim() == 5:
                image = image.squeeze(dim=1)  # Loại bỏ chiều dư thừa nếu có

            pred, _ = model(image)  # Bỏ qua unsqueeze(dim=0) nếu batch size đã được xử lý trong DataLoader
            pred = pred.squeeze(dim=0)  # Loại bỏ chiều batch nếu có
            
            # Nếu pred đã là numpy array, không cần gọi .cpu()
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            pred = pred.transpose(1, 2, 0)  # Chuyển đổi tensor PyTorch thành numpy array và thay đổi trật tự chiều
            pred = np.argmax(pred, axis=2).astype(np.uint8)

            # Tạo hình ảnh từ mảng dự đoán
            output_img = Image.fromarray(pred)

            parsing_result_path = os.path.join(args.output, img_name[:-4]+'.png')
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)

    return


def execute():
    get()