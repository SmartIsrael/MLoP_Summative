from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict_retrain import predict, retrain_model
from src.model import load_model

app = FastAPI()

model = load_model()

class PredictRequest(BaseModel):
    data: list

class RetrainRequest(BaseModel):
    data: list
    labels: list

@app.post('/predict')
def predict_endpoint(request: PredictRequest):
    try:
        # Create a DataFrame from the request data
        X_df = pd.DataFrame([request.data])
        
        # Make predictions
        predictions = predict(X_df)
        
        return {'predictions': predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/retrain')
async def retrain_endpoint(request: RetrainRequest):
    try:
        # Convert the request data to DataFrame
        X_df = pd.DataFrame(request.data)
        y_df = pd.Series(request.labels)

        # Ensure each row in X_df has 21 features
        def pad_or_truncate(row):
            if len(row) < 21:
                return row + [0.0] * (21 - len(row))
            elif len(row) > 21:
                return row[:21]
            return row

        X_df = X_df.apply(pad_or_truncate, axis=1)

        # Retrain the model
        global model  # Ensure that we update the global model variable
        model = retrain_model(X_df, y_df)

        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
