import os 
import gdown

os.makedirs('pre_trained', exist_ok=True)

files_to_download = {
    "1YI1_3JCYSsKyPNJQA9GSsr2EOk1e_S9w": "pre_trained/gmm_final.pth",
    "1Y7-I4_pK_jCzapzvidiZxKbRTLflPv8d": "pre_trained/model_ep_30.pth",
    "1DpDS0iGyny8J95JlkQ3EHz839xfL7LLr": "pre_trained/pose_deploy_linevec.prototxt",
    "1Ap2Kv0LnPSgi8Ag-jTizRvXR006QUNUt": "pre_trained/pose_iter_440000.caffemodel",
    "1mYATBsE9kcnbWgzIYeOEpxV4H6kNL19m": "pre_trained/PSPNet_last",
    "1PopD7nz007Mc9rIbLFKVZ09H74cWi5e6": "pre_trained/tom_final.pth",
}

for file_id, output_path in files_to_download.items():
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(url, output_path, quiet=False)