
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load DSID model data
df = pd.read_csv("dsid_model_data.csv")

@app.get("/predict")
def predict(nutrient: str = Query(...), label_claim: float = Query(...)):
    nutrient_lower = nutrient.strip().lower()

    # Attempt exact match first
    row = df[df['Nutrient'].str.lower() == nutrient_lower]

    # If not found, try partial match
    if row.empty:
        row = df[df['Nutrient'].str.lower().str.contains(nutrient_lower)]

    if row.empty:
        return {"error": "Nutrient not found in DSID model"}

    intercept = row['Pred_Intercept'].values[0]
    linear = row['Pred_Linear_Coeff'].values[0]
    quad = row['Pred_Quadratic_Coeff'].values[0]
    predicted = intercept + linear * label_claim + quad * (label_claim ** 2)
    return {
        "nutrient": row['Nutrient'].values[0],
        "label_claim": label_claim,
        "predicted_measured_amount": round(predicted, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
