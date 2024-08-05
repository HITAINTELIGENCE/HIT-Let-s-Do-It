from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uvicorn
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

@app.post('/upload/')
async def uploaded_images(
    user_name: str = Form(...),
    file_person: UploadFile = File(...),
    file_cloth: UploadFile = File(...)
):
    try:
        # Đọc ảnh từ file_person và lưu
        image_person = Image.open(io.BytesIO(file_person.file.read()))
        save_path_person = os.path.join(PERSON_DIR, file_person.filename)
        image_person.save(save_path_person)

        # Đọc ảnh từ file_cloth và lưu
        image_cloth = Image.open(io.BytesIO(file_cloth.file.read()))
        save_path_cloth = os.path.join(CLOTH_DIR, file_cloth.filename)
        image_cloth.save(save_path_cloth)

        # Xử lý mask cho áo
        execute_mask()

        # Phân tích pose
        pose_parse(file_person.filename)

        # Phân tích người
        execute()
        # Gắn áo vào cơ thể
        with open("./Database/val_pairs.txt", "w") as f:
                    f.write(file_person.filename  + " " + file_cloth.filename)

        predict()

         # Đường dẫn ảnh kết quả
        result_image_path = os.path.join(RESULT_DIR, f"{file_cloth.filename}")
        
        if os.path.exists(result_image_path):
            return FileResponse(result_image_path, media_type='image/jpeg', filename=f"{user_name}_result.jpg")
        else:
            return {"status": "Processing completed, but result image not found",
                    "test": result_image_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9000)