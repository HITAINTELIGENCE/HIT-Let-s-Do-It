from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uvicorn
import download_pre_trained
from try_on_clothes.utils.upscale import up_scale_x3_normal_fast
from cloth_mask.evaluate_mask import execute_mask
from pose_map.pose_parser_api import pose_parse
from human_parsing.evaluate_human_parsing import execute
from try_on_clothes.script import predict

app = FastAPI()

# Đường dẫn thư mục để lưu ảnh
PERSON_DIR = "./Database/val/person/"
CLOTH_DIR = "./Database/val/cloth/"
RESULT_DIR = "./Database/val/tryon-person"

os.makedirs(PERSON_DIR, exist_ok=True)
os.makedirs(CLOTH_DIR, exist_ok=True)

def change_extension(filename, new_extension):
    return os.path.splitext(filename)[0] + new_extension

@app.post('/upload/')
async def uploaded_images(
    user_name: str = Form(...),
    file_person: UploadFile = File(...),
    file_cloth: UploadFile = File(...)
):
    try:
        # Đọc ảnh từ file_person và lưu với đuôi .jpg
        image_person = Image.open(io.BytesIO(file_person.file.read()))
        person_filename_jpg = change_extension(file_person.filename, '.jpg')
        save_path_person = os.path.join(PERSON_DIR, person_filename_jpg)
        image_person = image_person.convert('RGB')  # Chuyển đổi sang RGB nếu ảnh có alpha channel
        image_person.save(save_path_person, format='JPEG')

        # Đọc ảnh từ file_cloth và lưu với đuôi .jpg
        image_cloth = Image.open(io.BytesIO(file_cloth.file.read()))
        image_cloth = image_cloth.resize((192, 256))
        cloth_filename_jpg = change_extension(file_cloth.filename, '.jpg')
        save_path_cloth = os.path.join(CLOTH_DIR, cloth_filename_jpg)
        image_cloth = image_cloth.convert('RGB')  # Chuyển đổi sang RGB nếu ảnh có alpha channel
        image_cloth.save(save_path_cloth, format='JPEG')

        # Xử lý mask cho áo
        execute_mask()

        # Phân tích pose
        pose_parse(person_filename_jpg)

        # Phân tích người
        execute()

        # Gắn áo vào cơ thể
        with open("./Database/val_pairs.txt", "w") as f:
            f.write(person_filename_jpg + " " + cloth_filename_jpg)

        predict()
        result = up_scale_x3_normal_fast("./Database/val/tryon-person/" + cloth_filename_jpg, "./Database/val/tryon-person/" + cloth_filename_jpg)

        result_image_path = os.path.join(RESULT_DIR, cloth_filename_jpg)
        
        if os.path.exists(result_image_path):
            return FileResponse(result_image_path, media_type='image/jpeg', filename=f"{user_name}_result.jpg")
        else:
            return {"status": "Processing completed, but result image not found",
                    "test": result_image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9000)
