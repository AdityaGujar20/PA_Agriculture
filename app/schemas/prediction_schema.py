from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    soil_pH: float
    soil_N: float
    soil_P: float
    rainfall_mm: float
    temp_avg: float
    fertilizer_kg_per_ha: float
    irrigation_mm: float
    pesticide_ml: float
    input_cost_total: float
    environmental_score: float
    month: int = Field(..., ge=1, le=12)
    day_of_year: int = Field(..., ge=1, le=366)
    crop_type: str
    year: int = Field(..., ge=2000, le=2100)
