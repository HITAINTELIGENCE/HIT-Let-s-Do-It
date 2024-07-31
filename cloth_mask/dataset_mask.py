import os
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# class VITONDataset(Dataset):
#     def __init__(self, root_dir, txt_file, transform=None):
#         self.root_dir = root_dir
#         self.txt_file = txt_file
#         self.transform = transform
#         self.img_path_lst = []
#         with open(self.txt_file) as file_in:
#             for line in file_in:
#                 self.img_path_lst.append(line)    
                
#     def __len__(self):
#         return len(self.img_path_lst)
    
#     def __getitem__(self, idx):
#         img_name = f"{self.img_path_lst[idx]}".strip()+".jpg"
#         image_path = os.path.join(self.root_dir, "cloth", img_name)
#         mask_path = os.path.join(self.root_dir, "cloth-mask", img_name)

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         # foreground -> 1
#         # background 2 -> 0
#         # 3 -> 1
#         mask[mask == 2] = 0
#         mask[mask == 1] = 1
#         # image (RGB), mask (2D matrix)
#         if self.transform is not None:
#             transformed = self.transform(image=image, mask=mask)
#             transformed_image = transformed['image']
#             transformed_mask = transformed['mask']
#         return transformed_image, transformed_mask
    
class VITONDataset(Dataset):
    def __init__(self, root_dir, input_size = [512, 512], transform=None):
        self.root_dir = root_dir
        self.input_size = input_size
        self.transform = transform
        self.input_size = np.asarray(input_size)

        self.file_list = os.listdir(self.root_dir)   
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, img_name

    

