from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, eda, preprocess, model_train, predict, optimize, shap

app = FastAPI(title="AgriPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(eda.router, prefix="/eda", tags=["EDA"])
app.include_router(preprocess.router, prefix="/preprocess", tags=["Preprocess"])
app.include_router(model_train.router, prefix="/train", tags=["Model Training"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(optimize.router, prefix="/optimize", tags=["Optimization"])
app.include_router(shap.router, prefix="/shap", tags=["SHAP"])
