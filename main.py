from fastapi import FastAPI, UploadFile
import torch
from src.model import CustomMobilenetV2
from src.upload import upload_img
from src.predict import pred_image

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def upload(file: UploadFile):
    try:
        file_location = upload_img(file)

        output_size = 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        path = "model/weights_best.pth"
        model = CustomMobilenetV2(output_size).to(device)
        weights = torch.load(path, map_location='cpu')
        model.load_state_dict(weights)

        predict_class = pred_image(file_location, model, device)

        [result, confident] = predict_class
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    return { "result": result, "confident": confident }