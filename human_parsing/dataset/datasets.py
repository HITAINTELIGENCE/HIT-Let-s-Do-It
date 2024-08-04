import os
import json
import cv2
import random
import numpy as np

import torch
from torch.utils import data
from PIL import Image


class HumanDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.input_size = np.asarray(input_size)

        self.file_list = os.listdir(self.root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        input = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        input = input.resize((192, 256), Image.LANCZOS)
    
        input = self.transform(input)

        return input, img_name
