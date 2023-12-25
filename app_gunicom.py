# main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
   


# Load the pre-trained RandomForest model
model_path = "models/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
    features = [input_data.Time,
                input_data.V1,
                input_data.V2,
                input_data.V3,
                input_data.V4,
                input_data.V5,
                input_data.V6,
                input_data.V7,
                input_data.V8,
                input_data.V9,
                input_data.V10,
                input_data.V11,
                
                ]
    prediction = model.predict([features])[0].item()
    # Return the prediction
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# CMD: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_gunicorn:app

# Uvicorn is a lightweight ASGI (Asynchronous Server Gateway Interface) server that specifically serves ASGI applications, such as those built with FastAPI.
# It is responsible for handling the asynchronous aspects of the application, making it efficient for high-concurrency scenarios.

# Gunicorn is a WSGI (Web Server Gateway Interface) server. While it is not designed for handling asynchronous tasks directly, it can be used to serve synchronous WSGI applications, including FastAPI applications.
# Gunicorn is a pre-fork worker model server, meaning it spawns multiple worker processes to handle incoming requests concurrently. Each worker runs in a separate process and can handle one request at a time.