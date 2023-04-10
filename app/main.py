import numpy as np
from io import BytesIO
from tensorflow import saved_model
from fastapi import FastAPI,File,UploadFile
from PIL import Image


app = FastAPI(title="Handwriting Classify")

# load ml model
clf = saved_model.load("mnist_clf") # /app/mnist_clf


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"

def preprocessing(image):
    image = Image.open(BytesIO(image)) # load bytes file to Image object
    image = image.convert("L") # Convert to greyscale 
    image = np.asarray(image.resize((28,28))) # covert image to numpy and resize to 28,28 ml model require input in 28x28
    image = image / 255 # normalize
    image = np.expand_dims(image,-1) # expand the last dimention make (28,28) to (28,28,1)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # data validation
    extension = file.filename.split(".")[-1] in ("jgp","jpeg","png")
    if not extension:
        return "Unsupport file extension"
    # data preprocessing
    image = preprocessing(await file.read())

    # make prediction
    pred = clf([image])
    pred = str(np.argmax(pred.numpy()))
    return {"Prediction": pred}



