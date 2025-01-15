from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict
import joblib
import pandas as pd

# Load the pre-trained ML model
model = joblib.load("RedWineQualityDetector.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_data(data: Dict):
    """Analyze the wine quality data using the pre-trained ML model."""
    try:
        # Convert the incoming data to a pandas DataFrame
        wine_samples = data.get('data', [])
        if not wine_samples:
            return JSONResponse(
                content={"error": "No data provided"},
                status_code=400
            )

        df = pd.DataFrame(wine_samples)

        # Ensure all required columns are present
        required_columns = [
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]

        missing_cols = [
            col for col in required_columns if col not in df.columns]
        if missing_cols:
            return JSONResponse(
                content={"error": f"Missing required columns: {missing_cols}"},
                status_code=400
            )

        # Ensure DataFrame only contains required columns in correct order
        df = df[required_columns]

        # Convert all values to float
        df = df.astype(float)

        # Predict quality using the loaded model
        predictions = model.predict(df)

        return {"results": predictions.tolist()}

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
