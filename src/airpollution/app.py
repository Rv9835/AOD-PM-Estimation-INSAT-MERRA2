from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from airpollution.config import load_config
from airpollution.data import load_dataset, preprocess_dataset, split_features_target
from airpollution.evaluation import compute_regression_metrics
from airpollution.join import SpatiotemporalJoiner, SpatiotemporalJoinError
from airpollution.logging_utils import setup_logging
from airpollution.modeling import predict, train_random_forest

setup_logging()

app = FastAPI(
    title="Air Pollution Monitoring API",
    description="Estimate PM2.5/PM10 from satellite AOD, ground stations, and meteorology",
    version="0.2.0",
)


class SourceFetchRequest(BaseModel):
    """Request to fetch unified data from CPCB, INSAT, MERRA-2."""

    start_date: datetime = Field(..., description="Start date (ISO format)")
    end_date: datetime = Field(..., description="End date (ISO format)")
    station_ids: list[str] | None = Field(
        default=None, description="Optional CPCB station IDs"
    )
    insat_radius_km: float = Field(
        default=10.0, description="INSAT search radius (km)"
    )
    merra2_radius_km: float = Field(
        default=50.0, description="MERRA-2 search radius (km)"
    )


class MetricsResponse(BaseModel):
    """Model evaluation metrics."""

    rmse: float
    mae: float
    r2: float


class PredictionResponse(BaseModel):
    """Prediction result."""

    predictions: list[float]
    station_ids: list[str] | None = None


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "airpollution-api"}


@app.post("/fetch-data", tags=["Data"])
async def fetch_unified_data(
    request: SourceFetchRequest,
) -> dict[str, Any]:
    """
    Fetch and join data from CPCB, INSAT, and MERRA-2.

    Returns unified spatiotemporal dataset ready for modeling.
    """
    try:
        joiner = SpatiotemporalJoiner()
        dataframe = joiner.fetch_and_join(
            request.start_date,
            request.end_date,
            station_ids=request.station_ids,
            insat_radius_km=request.insat_radius_km,
            merra2_radius_km=request.merra2_radius_km,
        )
        return {
            "status": "success",
            "rows": len(dataframe),
            "columns": dataframe.columns.tolist(),
            "date_range": {
                "start": dataframe["time"].min().isoformat(),
                "end": dataframe["time"].max().isoformat(),
            },
            "stations": dataframe["station_id"].unique().tolist()
            if "station_id" in dataframe.columns
            else [],
        }
    except SpatiotemporalJoinError as e:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/train", tags=["Model"])
async def train_model() -> MetricsResponse:
    """
    Train a new Random Forest model on cached dataset.

    Returns test set metrics.
    """
    try:
        config = load_config("configs/base.yaml")
        data = load_dataset(config)
        data = preprocess_dataset(data, config)
        x_train, x_test, y_train, y_test = split_features_target(data, config)

        model = train_random_forest(config, x_train, y_train)
        predictions = predict(model, x_test)
        metrics = compute_regression_metrics(y_test.to_numpy(), predictions.to_numpy())

        return MetricsResponse(rmse=metrics.rmse, mae=metrics.mae, r2=metrics.r2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", tags=["Model"])
async def predict_pm(features: dict[str, list[float]]) -> PredictionResponse:
    """
    Predict PM2.5 values for given feature vectors.

    Expects features dict with keys: aod, temperature, humidity, wind_speed,
    boundary_layer_height, lat, lon, day_of_year
    """
    try:
        import pandas as pd

        config = load_config("configs/base.yaml")

        # Load trained model
        import joblib

        estimator = joblib.load(config.paths.model_path)

        # Build feature dataframe
        feature_df = pd.DataFrame(features)
        required = config.model.feature_columns
        missing = [col for col in required if col not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        # Predict
        preds = estimator.predict(feature_df[required])

        return PredictionResponse(predictions=preds.tolist())
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Train first with POST /train",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Model"])
async def get_metrics() -> MetricsResponse:
    """Get cached model evaluation metrics."""
    try:
        import json

        from airpollution.config import load_config

        config = load_config("configs/base.yaml")
        metrics_dict = json.loads(config.paths.metrics_path.read_text(encoding="utf-8"))
        return MetricsResponse(**metrics_dict)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Train first with POST /train",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "service": "Air Pollution Monitoring API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
    }
