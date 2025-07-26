
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

# Load model
df = pd.read_csv("dsid_model_with_age.csv")

@app.get("/predict")
def predict(nutrient: str = Query(...), label_claim: float = Query(...), age_group: str = Query("Adult")):
    nutrient_lower = nutrient.strip().lower()
    age_group = age_group.strip().capitalize()

    # Filter by Age Group
    df_group = df[df['Age_Group'] == age_group]

    # Try strict match first
    exact_match = df_group[df_group['Nutrient'].str.lower() == nutrient_lower]
    if not exact_match.empty:
        row = exact_match.iloc[0]
    else:
        # Try fuzzy match
        fuzzy_match = df_group[df_group['Nutrient'].str.lower().str.contains(nutrient_lower)]
        if fuzzy_match.empty:
            return {"error": f"Nutrient not found in DSID model for {age_group} group."}
        row = fuzzy_match.iloc[0]

    intercept = row['Pred_Intercept']
    linear = row['Pred_Linear_Coeff']
    quad = row['Pred_Quadratic_Coeff']
    predicted = intercept + linear * label_claim + quad * (label_claim ** 2)

    return {{
        "nutrient": row['Nutrient'],
        "age_group": row['Age_Group'],
        "label_claim": label_claim,
        "predicted_measured_amount": round(predicted, 4),
        "model_intercept": round(intercept, 4),
        "model_linear": round(linear, 4),
        "model_quadratic": round(quad, 6)
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
