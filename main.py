from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import joblib
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model with error handling
MODEL_PATH = "random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found!")

model = joblib.load(MODEL_PATH)

@app.post("/predict-loan-csv")
async def predict_loan_csv(file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty!")

        # Extract features (ensure columns match the model input)
        feature_columns = df.columns.tolist()
        X = df.values  

        # Check if input matches model expectation
        if X.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must have {model.n_features_in_} features, but received {X.shape[1]}"
            )

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of rejection

        # Add results to DataFrame
        df["loan_approval"] = predictions  # 0 = Approved, 1 = Not Approved
        df["approval_probability"] = probabilities

        # Convert result to JSON
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )
