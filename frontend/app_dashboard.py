"""
Air Pollution Prediction Dashboard
Non-conventional 3D Futuristic Design with Glassmorphism and Pydeck 3D Mapping
"""

import sys
from pathlib import Path

# Load .env file for local development (python-dotenv auto-discovery)
try:
    from dotenv import load_dotenv
    PROJECT_ROOT_TEMP = Path(__file__).resolve().parent.parent
    env_file = PROJECT_ROOT_TEMP / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not required in production

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = str(PROJECT_ROOT / "backend")
BACKEND_SRC_PATH = str(PROJECT_ROOT / "backend" / "src")

if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)
if BACKEND_SRC_PATH not in sys.path:
    sys.path.insert(0, BACKEND_SRC_PATH)

CONFIG_PATH = PROJECT_ROOT / "backend" / "configs" / "base.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"


def resolve_project_path(path_like: str) -> Path:
    """Resolve path to absolute path anchored at project root when needed."""
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import yaml

from airpollution.cities import CityManager, haversine_distance, bearing_between_cities
from airpollution.multi_city_data import MultiCityDataLoader
from airpollution.models.factory import ModelFactory
from airpollution.evaluators import CrossCityEvaluator

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="🛰️ Air Pollution Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# CUSTOM CSS: CINEMATIC DARK MODE + GLASSMORPHISM
# =====================================================
def inject_custom_css() -> None:
    """Inject custom CSS for futuristic dark theme with glassmorphism."""
    css = """
    <style>
        /* Font imports */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

        /* Root colors: Deep space black + neon accents */
        :root {
            --bg-dark: #0a0e27;
            --bg-darker: #050812;
            --text-light: #e0e6ff;
            --text-dim: #8892b0;
            --accent-cyan: #00d9ff;
            --accent-magenta: #ff006e;
            --accent-electric-blue: #0080ff;
            --accent-purple: #7c3aed;
            --glass-bg: rgba(15, 23, 42, 0.7);
            --glass-border: rgba(0, 217, 255, 0.2);
        }

        /* Main background */
        body, .stApp {
            background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%);
            color: var(--text-light);
            font-family: 'Space Mono', monospace;
        }

        /* Remove default Streamlit header/footer */
        header {
            background: transparent !important;
            border: none !important;
        }

        footer {
            display: none !important;
        }

        /* Main container */
        .main {
            background: transparent !important;
            padding: 20px 40px;
        }

        /* Sidebar: Dark glassmorphic panel */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--glass-bg) 0%, rgba(20, 30, 50, 0.5) 100%);
            border-left: 2px solid var(--accent-cyan);
            backdrop-filter: blur(10px);
        }

        /* Cards: Glassmorphic floating design */
        .st-emotion-cache-uf99v0, .st-emotion-cache-1avcm0n, section {
            background: linear-gradient(135deg, var(--glass-bg) 0%, rgba(0, 217, 255, 0.05) 100%);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            padding: 20px !important;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
            transition: all 0.3s ease;
        }

        .st-emotion-cache-uf99v0:hover, section:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 8px 32px rgba(0, 217, 255, 0.25);
            transform: translateY(-2px);
        }

        /* Tabs: Neon accent */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 2px solid var(--text-dim);
            color: var(--text-dim);
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .stTabs [aria-selected="true"] [data-baseweb="tab"] {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-electric-blue));
            color: var(--bg-dark);
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        }

        /* Headings: Futuristic Orbitron font */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', sans-serif;
            color: var(--accent-cyan);
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
            letter-spacing: 2px;
            font-weight: 700;
        }

        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        h2 {
            font-size: 1.8rem;
            color: var(--accent-electric-blue);
        }

        /* Metric cards animation */
        @keyframes pulse-glow {
            0%, 100% {
                box-shadow: 0 0 5px rgba(0, 217, 255, 0.3), inset 0 0 5px rgba(0, 217, 255, 0.1);
            }
            50% {
                box-shadow: 0 0 20px rgba(0, 217, 255, 0.6), inset 0 0 10px rgba(0, 217, 255, 0.2);
            }
        }

        @keyframes slide-up {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .metric-card {
            animation: pulse-glow 3s ease-in-out infinite, slide-up 0.5s ease-out;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(0, 128, 255, 0.05));
            border: 2px solid var(--accent-cyan);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 0 15px rgba(0, 217, 255, 0.2);
        }

        /* Buttons: Neon glow */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-electric-blue));
            color: var(--bg-dark);
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            box-shadow: 0 0 15px rgba(0, 217, 255, 0.4);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            box-shadow: 0 0 30px rgba(0, 217, 255, 0.8);
            transform: scale(1.05);
        }

        /* Text styling */
        .stMarkdown, .stSubheader {
            color: var(--text-light);
        }

        /* Divider: Neon line */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
            margin: 30px 0;
        }

        /* Loading spinner enhancement */
        .stSpinner > div {
            border-color: var(--accent-cyan) !important;
            border-right-color: transparent !important;
        }

        /* Selectbox styling */
        [data-baseweb="select"] {
            border: 2px solid var(--accent-cyan) !important;
            background: var(--glass-bg) !important;
            color: var(--text-light) !important;
        }

        /* Success/Error messages */
        .stSuccess {
            background: rgba(0, 217, 255, 0.1) !important;
            border-left: 4px solid var(--accent-cyan) !important;
        }

        .stError {
            background: rgba(255, 0, 110, 0.1) !important;
            border-left: 4px solid var(--accent-magenta) !important;
        }

        .stWarning {
            background: rgba(255, 165, 0, 0.1) !important;
            border-left: 4px solid #ffa500 !important;
        }

        /* Plotly chart container */
        .plotly-graph-div {
            background: var(--glass-bg) !important;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        }

        /* Data table styling */
        .dataframe {
            background: var(--glass-bg) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 8px;
        }

        table {
            background: var(--glass-bg) !important;
        }

        th {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-electric-blue)) !important;
            color: var(--bg-dark) !important;
            font-weight: 700;
        }

        td {
            background: var(--glass-bg) !important;
            color: var(--text-light) !important;
            border-bottom: 1px solid var(--glass-border) !important;
        }

        /* Floating animation for hero elements */
        @keyframes float {
            0%, 100% {
                transform: translateY(-5px);
            }
            50% {
                transform: translateY(5px);
            }
        }

        .float-element {
            animation: float 4s ease-in-out infinite;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =====================================================
# HELPER: LOAD CONFIGURATIONS AND DATA
# =====================================================
@st.cache_resource
def load_config(config_path: str = str(CONFIG_PATH)) -> Dict:
    """Load YAML configuration."""
    with open(resolve_project_path(config_path), "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_data_loader(config_path: str = str(CONFIG_PATH)) -> MultiCityDataLoader:
    """Load MultiCityDataLoader."""
    return MultiCityDataLoader(str(resolve_project_path(config_path)))


@st.cache_resource
def load_city_manager() -> CityManager:
    """Load CityManager."""
    return CityManager()


@st.cache_data
def load_scenario_data(scenario_name: str, _data_loader: MultiCityDataLoader) -> Tuple:
    """Load and cache scenario data."""
    try:
        X_train, y_train, X_test, y_test = _data_loader.split_by_scenario(scenario_name)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load scenario {scenario_name}: {e}")
        st.error(f"Failed to load scenario: {e}")
        return None, None, None, None


@st.cache_resource
def load_model(scenario_name: str, model_name: str, model_dir: str):
    """Load a trained model from disk."""
    model_root = resolve_project_path(model_dir)

    model_path = model_root / scenario_name / model_name / "model.joblib"
    
    if not model_path.exists():
        return None
    
    try:
        model_class_map = {
            "linear_regression": "LinearRegressionRegressor",
            "random_forest": "RandomForestRegressor",
            "xgboost": "XGBoostRegressor",
            "lightgbm": "LightGBMRegressor",
            "neural_network": "MLPNeuralNetworkRegressor",
        }
        
        from airpollution.models.factory import (
            LinearRegressionRegressor,
            RandomForestRegressor,
            XGBoostRegressor,
            LightGBMRegressor,
            MLPNeuralNetworkRegressor,
        )
        
        classes = {
            "LinearRegressionRegressor": LinearRegressionRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "XGBoostRegressor": XGBoostRegressor,
            "LightGBMRegressor": LightGBMRegressor,
            "MLPNeuralNetworkRegressor": MLPNeuralNetworkRegressor,
        }
        
        model_cls_name = model_class_map.get(model_name)
        if model_cls_name:
            model_cls = classes[model_cls_name]
            return model_cls.load(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model {scenario_name}/{model_name}: {e}")
    
    return None


# =====================================================
# TAB 1: COMMAND CENTER - HOME WITH ANIMATED METRICS
# =====================================================
def render_command_center(config: Dict, data_loader: MultiCityDataLoader) -> None:
    """Render Command Center tab with animated metric cards."""
    st.markdown("### 🛰️ **COMMAND CENTER** - System Status")
    st.markdown("Real-time monitoring of multi-city air pollution prediction pipeline")

    scenarios = config.get("training_scenarios", [])
    enabled_models = [
        algo["name"]
        for algo in config.get("models", {}).get("algorithms", [])
        if algo.get("enabled", True)
    ]
    expected_models = len(scenarios) * len(enabled_models)

    model_root = resolve_project_path(config["artifacts"]["models_dir"])
    trained_models = len(list(model_root.glob("*/*/model.joblib"))) if model_root.exists() else 0
    training_coverage = (trained_models / expected_models) if expected_models else 0.0

    metrics_path = resolve_project_path(config["artifacts"]["metrics_dir"]) / "comparison_results.json"
    metrics_data = {"scenarios": []}
    evaluated_runs = 0
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            evaluated_runs = sum(
                len(s.get("models", [])) for s in metrics_data.get("scenarios", [])
            )
        except Exception:
            evaluated_runs = 0

    if training_coverage >= 0.95 and evaluated_runs > 0:
        status_emoji = "🟢"
        status_text = "Operational"
    elif training_coverage >= 0.5:
        status_emoji = "🟡"
        status_text = "Partially Ready"
    else:
        status_emoji = "🔴"
        status_text = "Needs Setup"
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics
    with col1:
        st.markdown(
            '<div class="metric-card">'
            f'<h3 style="color: var(--accent-cyan);">CITIES</h3>'
            f'<h2 style="color: var(--accent-electric-blue); font-size: 2.5rem;">{len(config.get("data", {}).get("cities_to_use", []))}</h2>'
            '<p style="color: var(--text-dim);">Active Monitoring</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            '<div class="metric-card">'
            f'<h3 style="color: var(--accent-cyan);">TRAINED</h3>'
            f'<h2 style="color: var(--accent-electric-blue); font-size: 2.5rem;">{trained_models}/{expected_models}</h2>'
            '<p style="color: var(--text-dim);">Model Artifacts</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            '<div class="metric-card">'
            f'<h3 style="color: var(--accent-cyan);">EVALUATIONS</h3>'
            f'<h2 style="color: var(--accent-electric-blue); font-size: 2.5rem;">{evaluated_runs}</h2>'
            '<p style="color: var(--text-dim);">Completed Runs</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    
    with col4:
        st.markdown(
            '<div class="metric-card">'
            f'<h3 style="color: var(--accent-cyan);">STATUS</h3>'
            f'<h2 style="color: #00ff00; font-size: 2.5rem;">{status_emoji}</h2>'
            f'<p style="color: var(--text-dim);">{status_text}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.progress(training_coverage, text=f"Training readiness: {training_coverage * 100:.1f}%")
    st.caption("Use this tab to verify artifact health before running diagnostics, comparisons, or map transfer analysis.")
    
    st.markdown("---")
    
    # System overview
    st.markdown("### 📊 **SYSTEM ARCHITECTURE**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🏙️ **Cities Registry**
        - **Delhi**: National Capital, Metro area, Altitude 216m
        - **Mumbai**: Coastal Metropolitan, Altitude 14m
        - **Bangalore**: High Altitude City, Altitude 920m
        - **Kolkata**: Eastern Metro, Gangetic Plain, Altitude 9m
        - **Hyderabad**: Deccan Plateau, Altitude 505m
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 **Model Ensemble**
        1. **Linear Regression** - Baseline interpretability
        2. **Random Forest** - Non-linear patterns
        3. **XGBoost** - Gradient boosting power
        4. **LightGBM** - Fast inference
        5. **Neural Network** - Deep learning capability
        """)
    
    st.markdown("---")
    
    # Quick scenario selector
    st.markdown("### 🎯 **QUICK NAVIGATION**")
    scenarios = config.get("training_scenarios", [])
    scenario_names = [s["name"] for s in scenarios]
    
    selected_scenario = st.selectbox(
        "Select Scenario to Analyze",
        scenario_names,
        key="scenario_select_command",
    )
    
    if selected_scenario:
        scenario_desc = next(
            (s.get("description", "") for s in scenarios if s["name"] == selected_scenario),
            "No description available",
        )
        st.info(f"📝 {scenario_desc}")

        scenario_metrics = next(
            (item for item in metrics_data.get("scenarios", []) if item.get("name") == selected_scenario),
            None,
        )

        if scenario_metrics and scenario_metrics.get("models"):
            ranked_models = sorted(
                scenario_metrics["models"],
                key=lambda model_item: (
                    model_item.get("test", {}).get("r2", float("-inf")),
                    -model_item.get("test", {}).get("rmse", float("inf")),
                ),
                reverse=True,
            )
            best_model = ranked_models[0]

            st.markdown("#### 🏆 **BEST MODEL INSIGHT**")
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            with insight_col1:
                st.metric("Best Model", best_model.get("name", "N/A").replace("_", " ").title())
            with insight_col2:
                st.metric("Test R²", f"{best_model.get('test', {}).get('r2', 0.0):.4f}")
            with insight_col3:
                st.metric("Robustness", f"{best_model.get('robustness_score', 0.0):.2f}/100")

            st.caption(
                f"Quick read: RMSE {best_model.get('test', {}).get('rmse', 0.0):.2f} | "
                f"MAE {best_model.get('test', {}).get('mae', 0.0):.2f}. "
                "Use Neural Analysis for residual behavior and Holographic Comparisons for full ranking."
            )
        else:
            st.caption("Run evaluation to unlock scenario-specific best model insights in Command Center.")


# =====================================================
# TAB 2: NEURAL ANALYSIS - 3D RESIDUALS AND DIAGNOSTICS
# =====================================================
def render_neural_analysis(config: Dict, data_loader: MultiCityDataLoader) -> None:
    """Render 3D neural analysis with model residuals."""
    st.markdown("### 🧠 **NEURAL ANALYSIS** - Model Diagnostics in 3D")
    st.caption("Workflow: pick a scenario + model, click Analyze, then inspect residual clouds and distribution plots.")
    
    scenarios = config.get("training_scenarios", [])
    scenario_names = [s["name"] for s in scenarios]
    model_dir = config["artifacts"]["models_dir"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario = st.selectbox("Select Scenario", scenario_names, key="na_scenario")
    
    with col2:
        algorithms = [
            algo["name"]
            for algo in config["models"]["algorithms"]
            if algo.get("enabled", True)
        ]
        model = st.selectbox("Select Model", algorithms, key="na_model")
    
    with col3:
        if st.button("🔍 Analyze", key="na_analyze"):
            st.session_state.analyze_neural = True
    
    if st.session_state.get("analyze_neural", False):
        with st.spinner("Loading model and generating predictions..."):
            # Load data
            X_train, y_train, X_test, y_test = load_scenario_data(scenario, data_loader)
            
            if X_train is None:
                st.error("Failed to load data")
                return

            model_root = resolve_project_path(model_dir)
            checked_model_path = model_root / scenario / model / "model.joblib"

            if not checked_model_path.exists():
                st.warning(
                    f"Model artifacts not found at {checked_model_path}. "
                    "Please run the training script for this scenario first."
                )
                return
            
            # Load model
            trained_model = load_model(scenario, model, model_dir)
            if trained_model is None:
                st.warning(
                    f"Model artifacts not found at {checked_model_path}. "
                    "Please run the training script for this scenario first."
                )
                return
            
            # Generate predictions
            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)
            
            # Compute residuals
            train_residuals = y_train.values - y_train_pred
            test_residuals = y_test.values - y_test_pred
            
            # Create 3D scatter plot: Actual vs Predicted vs Residuals
            fig = go.Figure()
            
            # Test set points
            fig.add_trace(
                go.Scatter3d(
                    x=y_test.values,
                    y=y_test_pred,
                    z=test_residuals,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=test_residuals,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Residuals"),
                    ),
                    name="Test Data",
                    text=[f"Actual: {a:.2f}<br>Pred: {p:.2f}<br>Residual: {r:.2f}" 
                          for a, p, r in zip(y_test.values, y_test_pred, test_residuals)],
                    hovertemplate="<b>Test Point</b><br>%{text}<extra></extra>",
                )
            )
            
            # Perfect prediction line (y=x)
            max_val = max(y_test.max(), y_test_pred.max())
            min_val = min(y_test.min(), y_test_pred.min())
            fig.add_trace(
                go.Scatter3d(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    z=[0, 0],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(color="cyan", width=3),
                    hoverinfo="skip",
                )
            )
            
            fig.update_layout(
                title=f"<b>3D Model Analysis: {model} on {scenario}</b><br><sub>X: Actual PM2.5 | Y: Predicted PM2.5 | Z: Residuals</sub>",
                scene=dict(
                    xaxis_title="Actual PM2.5",
                    yaxis_title="Predicted PM2.5",
                    zaxis_title="Residuals",
                    bgcolor="rgba(10, 14, 39, 0.9)",
                    xaxis=dict(backgroundcolor="rgba(0, 217, 255, 0.1)", gridcolor="rgba(0, 217, 255, 0.2)"),
                    yaxis=dict(backgroundcolor="rgba(0, 217, 255, 0.1)", gridcolor="rgba(0, 217, 255, 0.2)"),
                    zaxis=dict(backgroundcolor="rgba(0, 217, 255, 0.1)", gridcolor="rgba(0, 217, 255, 0.2)"),
                ),
                paper_bgcolor="rgba(10, 14, 39, 0.8)",
                font=dict(color="rgba(224, 230, 255, 0.9)", family="Space Mono"),
                height=700,
                hovermode="closest",
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of residuals
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=test_residuals,
                        nbinsx=30,
                        name="Residuals",
                        marker_color="rgba(0, 217, 255, 0.7)",
                    )
                )
                fig_hist.update_layout(
                    title=f"<b>Residual Distribution</b>",
                    xaxis_title="Residual Value",
                    yaxis_title="Frequency",
                    paper_bgcolor="rgba(10, 14, 39, 0.8)",
                    plot_bgcolor="rgba(15, 23, 42, 0.6)",
                    font=dict(color="rgba(224, 230, 255, 0.9)"),
                    height=400,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Q-Q plot (quantiles)
                from scipy import stats
                sorted_residuals = np.sort(test_residuals)
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
                
                fig_qq = go.Figure()
                fig_qq.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_residuals,
                        mode="markers",
                        marker=dict(color="rgba(0, 217, 255, 0.7)", size=6),
                        name="Residuals",
                    )
                )
                # Add diagonal reference
                min_q = theoretical_quantiles.min()
                max_q = theoretical_quantiles.max()
                fig_qq.add_trace(
                    go.Scatter(
                        x=[min_q, max_q],
                        y=[min_q, max_q],
                        mode="lines",
                        line=dict(color="rgba(255, 0, 110, 0.5)", dash="dash"),
                        name="Normal Distribution",
                    )
                )
                
                fig_qq.update_layout(
                    title="<b>Q-Q Plot</b>",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    paper_bgcolor="rgba(10, 14, 39, 0.8)",
                    plot_bgcolor="rgba(15, 23, 42, 0.6)",
                    font=dict(color="rgba(224, 230, 255, 0.9)"),
                    height=400,
                )
                st.plotly_chart(fig_qq, use_container_width=True)


# =====================================================
# TAB 3: HOLOGRAPHIC COMPARISONS - RADAR & BAR CHARTS
# =====================================================
def render_holographic_comparisons(config: Dict) -> None:
    """Render model comparisons with radar and bar charts."""
    st.markdown("### 📊 **HOLOGRAPHIC COMPARISONS** - Model Performance Metrics")
    st.caption("Use this tab to compare model generalization: higher R² and robustness with lower RMSE/MAE is better.")
    
    scenarios = config.get("training_scenarios", [])
    scenario_names = [s["name"] for s in scenarios]
    metrics_dir = resolve_project_path(config["artifacts"]["metrics_dir"])
    
    selected_scenario = st.selectbox(
        "Select Scenario for Comparison",
        scenario_names,
        key="holo_scenario",
    )
    
    # Try to load metrics JSON
    metrics_json_path = metrics_dir / "comparison_results.json"
    
    if not metrics_json_path.exists():
        st.warning("📊 No comparison results found. Run evaluation pipeline first.")
        return
    
    import json
    with open(metrics_json_path, "r") as f:
        all_results = json.load(f)
    
    # Find scenario in results
    scenario_results = None
    for scenario_data in all_results.get("scenarios", []):
        if scenario_data["name"] == selected_scenario:
            scenario_results = scenario_data
            break
    
    if scenario_results is None:
        st.warning(f"No results found for scenario: {selected_scenario}")
        return
    
    # Extract model data
    models_data = scenario_results["models"]
    
    # Create radar chart for robustness comparison
    fig_radar = go.Figure()
    
    for model_data in models_data:
        model_name = model_data["name"]
        
        # Normalize metrics to 0-1 scale
        test_rmse = model_data["test"]["rmse"]
        test_r2 = max(0, model_data["test"]["r2"])  # Clamp negative R²
        robustness = model_data["robustness_score"] / 100.0
        mae = model_data["test"]["mae"]
        
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[test_r2, robustness, 1 - min(test_rmse / 100, 1), mae / 100],
                theta=["R² (Test)", "Robustness", "RMSE Score", "MAE Score"],
                fill="toself",
                name=model_name,
                opacity=0.7,
            )
        )
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickcolor="rgba(0, 217, 255, 0.3)",
            ),
            bgcolor="rgba(15, 23, 42, 0.3)",
        ),
        paper_bgcolor="rgba(10, 14, 39, 0.8)",
        font=dict(color="rgba(224, 230, 255, 0.9)", family="Space Mono"),
        title=f"<b>Model Performance Radar: {selected_scenario}</b>",
        height=500,
        hovermode="closest",
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Bar chart: Robustness scores
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar_robustness = go.Figure()
        
        model_names = [m["name"] for m in models_data]
        robustness_scores = [m["robustness_score"] for m in models_data]
        
        fig_bar_robustness.add_trace(
            go.Bar(
                x=model_names,
                y=robustness_scores,
                marker=dict(
                    color=robustness_scores,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Score"),
                ),
                name="Robustness",
            )
        )
        
        fig_bar_robustness.update_layout(
            title="<b>Robustness Score Comparison</b>",
            xaxis_title="Model",
            yaxis_title="Robustness (0-100)",
            paper_bgcolor="rgba(10, 14, 39, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="rgba(224, 230, 255, 0.9)"),
            height=400,
        )
        
        st.plotly_chart(fig_bar_robustness, use_container_width=True)
    
    with col2:
        fig_bar_rmse = go.Figure()
        
        test_rmses = [m["test"]["rmse"] for m in models_data]
        
        fig_bar_rmse.add_trace(
            go.Bar(
                x=model_names,
                y=test_rmses,
                marker=dict(
                    color=test_rmses,
                    colorscale="Reds_r",
                    showscale=True,
                    colorbar=dict(title="RMSE"),
                ),
                name="Test RMSE",
            )
        )
        
        fig_bar_rmse.update_layout(
            title="<b>Test RMSE Comparison (Lower is Better)</b>",
            xaxis_title="Model",
            yaxis_title="RMSE",
            paper_bgcolor="rgba(10, 14, 39, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="rgba(224, 230, 255, 0.9)"),
            height=400,
        )
        
        st.plotly_chart(fig_bar_rmse, use_container_width=True)
    
    # Detailed table
    st.markdown("### 📋 **DETAILED METRICS TABLE**")
    
    table_data = []
    for model_data in models_data:
        table_data.append({
            "Model": model_data["name"],
            "Train RMSE": f"{model_data['train']['rmse']:.4f}",
            "Test RMSE": f"{model_data['test']['rmse']:.4f}",
            "Train R²": f"{model_data['train']['r2']:.4f}",
            "Test R²": f"{model_data['test']['r2']:.4f}",
            "Robustness": f"{model_data['robustness_score']:.2f}/100",
            "RMSE Ratio": f"{model_data['rmse_ratio']:.2f}x",
        })
    
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True)


# =====================================================
# TAB 4: PLANETARY NETWORK - 3D PYDECK MAP
# =====================================================
def render_planetary_network(config: Dict) -> None:
    """Render 3D map with city pillars and robustness arcs."""
    st.markdown("### 🌍 **PLANETARY NETWORK** - Multi-City Robustness Topology")
    st.markdown("Training cities (blue pillars) → Test cities (red pillars) | Arc brightness = robustness transfer")
    st.caption("Interpretation: thicker/brighter arcs indicate stronger cross-city transfer robustness.")
    
    # Load metrics
    metrics_dir = resolve_project_path(config["artifacts"]["metrics_dir"])
    metrics_json_path = metrics_dir / "comparison_results.json"
    
    if not metrics_json_path.exists():
        st.warning("📊 No comparison results found. Run evaluation pipeline first.")
        return
    
    import json
    with open(metrics_json_path, "r") as f:
        all_results = json.load(f)
    
    city_manager = load_city_manager()
    cities_dict = city_manager.get_all_cities()
    
    # Build map layers
    layers = []
    
    # Layer 1: City pillars (hexagons)
    city_coordinates = []
    city_names = []
    city_colors = []
    city_heights = []
    
    # Get all scenarios to determine which cities are training vs test
    training_cities = set()
    test_cities = set()
    
    for scenario_data in all_results.get("scenarios", []):
        # Parse scenario name to extract cities
        scenario_name = scenario_data["name"]
        
        # Simple heuristic: if "to_" in name, extract training and test city
        if "_to_" in scenario_name:
            parts = scenario_name.split("_to_")
            if len(parts) == 2:
                train_part = parts[0]
                test_part = parts[1]
                
                # Extract city names
                for city_key in city_manager.list_cities():
                    if city_key in train_part:
                        training_cities.add(city_key)
                    if city_key in test_part:
                        test_cities.add(city_key)
    
    # Add all cities to map
    for city_key, city_config in cities_dict.items():
        city_coordinates.append([city_config.center_lon, city_config.center_lat])
        city_names.append(city_config.display_name)

        # Color based on role
        if city_key in training_cities and city_key in test_cities:
            city_colors.append([100, 200, 255])  # Purple for cities in both roles
            city_heights.append(5000)
        elif city_key in training_cities:
            city_colors.append([0, 150, 255])  # Blue for training cities
            city_heights.append(4000)
        elif city_key in test_cities:
            city_colors.append([255, 0, 110])  # Red for test cities
            city_heights.append(4000)
        else:
            city_colors.append([100, 100, 100])  # Gray for inactive
            city_heights.append(2000)

    city_df = pd.DataFrame(
        {
            "lat": [c[1] for c in city_coordinates],
            "lng": [c[0] for c in city_coordinates],
            "name": city_names,
            "height": city_heights,
            "color": city_colors,
        }
    )
    city_df["lat"] = pd.to_numeric(city_df["lat"], errors="coerce")
    city_df["lng"] = pd.to_numeric(city_df["lng"], errors="coerce")
    city_df["height"] = pd.to_numeric(city_df["height"], errors="coerce")
    city_df = city_df.dropna(subset=["lat", "lng", "height"])

    if city_df.empty:
        st.warning("⚠ Unable to render map: no valid city coordinates available.")
        return

    # Layer 1: Lightweight city halo layer for visibility
    city_halo_layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_df,
        get_position=["lng", "lat"],
        get_radius=85000,
        get_fill_color=[0, 217, 255, 70],
        get_line_color=[0, 217, 255, 170],
        stroked=True,
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )
    layers.append(city_halo_layer)

    # Layer 2: 3D city pillars
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=city_df,
        get_position=["lng", "lat"],
        get_elevation="height",
        elevation_scale=1000,
        radius=25000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    layers.append(column_layer)
    
    # Layer 3: 3D Arcs connecting cities (robustness transfer)
    arc_data = []
    
    for scenario_data in all_results.get("scenarios", []):
        scenario_name = scenario_data["name"]
        
        # Extract training and test cities from scenario name
        if "_to_" in scenario_name:
            parts = scenario_name.split("_to_")
            if len(parts) == 2:
                train_name = parts[0]
                test_name = parts[1]
                
                # Get robustness score (average across models)
                robustness_scores = [m["robustness_score"] for m in scenario_data["models"]]
                avg_robustness = np.mean(robustness_scores) if robustness_scores else 50
                
                # Map scenario names to cities
                train_city = None
                test_city = None
                
                for city_key in city_manager.list_cities():
                    if city_key in train_name:
                        train_city = city_key
                    if city_key in test_name:
                        test_city = city_key
                
                if train_city and test_city:
                    train_config = cities_dict[train_city]
                    test_config = cities_dict[test_city]
                    
                    # Arc color based on robustness (bright if high robustness)
                    arc_strength = avg_robustness / 100.0
                    arc_color = [
                        int(0 + 255 * arc_strength),  # Red component
                        int(217 * arc_strength),  # Green component
                        int(255 * (1 - arc_strength)),  # Blue component
                        int(150 * arc_strength),  # Alpha
                    ]
                    
                    arc_data.append({
                        "source_lat": train_config.center_lat,
                        "source_lng": train_config.center_lon,
                        "target_lat": test_config.center_lat,
                        "target_lng": test_config.center_lon,
                        "robustness": avg_robustness,
                        "color": arc_color,
                    })
    
    if arc_data:
        arc_df = pd.DataFrame(arc_data)
        for coord_col in ["source_lat", "source_lng", "target_lat", "target_lng", "robustness"]:
            arc_df[coord_col] = pd.to_numeric(arc_df[coord_col], errors="coerce")
        arc_df = arc_df.dropna(subset=["source_lat", "source_lng", "target_lat", "target_lng", "robustness"])
        
        if not arc_df.empty:
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arc_df,
                get_source_position=["source_lng", "source_lat"],
                get_target_position=["target_lng", "target_lat"],
                get_source_color=[0, 150, 255],
                get_target_color="color",
                auto_highlight=True,
                get_width=3,
                pickable=True,
            )

            layers.append(arc_layer)
    
    # Create map
    view_state = pdk.ViewState(
        longitude=78.0,
        latitude=22.0,
        zoom=4.0,
        pitch=45,
        bearing=0,
    )
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark",
        tooltip=True,
    )
    
    st.pydeck_chart(r)
    
    st.markdown("---")
    st.markdown("""
    ### 🗺️ **Map Legend**
    - 🔵 **Blue Pillars**: Training Cities
    - 🔴 **Red Pillars**: Test Cities
    - 🟣 **Purple Pillars**: Cities used in both roles
    - **Arc Color**: Green = High Robustness | Red = Low Robustness
    - **Arc Thickness**: Proportional to robustness transfer score
    - **Cyan Halo**: City influence footprint for visibility
    """)


# =====================================================
# TAB 5: PREDICTION ENGINE - REAL-TIME PM2.5 FORECASTING
# =====================================================
def render_prediction_engine(config: Dict, data_loader: MultiCityDataLoader) -> None:
    """Render interactive prediction interface with scenario-based forecasting."""
    st.markdown("### 🔮 **PREDICTION ENGINE** - Real-Time PM2.5 Forecasting")
    st.markdown("Input meteorological and satellite data to predict PM2.5 levels across cities")
    
    # Initialize session state for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Satellite & Meteorological Data")
        
        aod = st.slider(
            "Aerosol Optical Depth (AOD)",
            min_value=0.0,
            max_value=1.0,
            value=0.28,
            step=0.01,
            help="Lower values = clearer air, higher = more aerosols"
        )
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=25.0,
            step=0.5,
            help="Ambient temperature in degrees Celsius"
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=55,
            step=5,
            help="Relative humidity percentage"
        )
        
        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=10.0,
            value=3.2,
            step=0.1,
            help="Higher wind = better dispersion of pollutants"
        )
    
    with col2:
        st.subheader("🌍 Location & Temporal Data")
        
        boundary_layer_height = st.slider(
            "Boundary Layer Height (m)",
            min_value=300.0,
            max_value=2000.0,
            value=780.0,
            step=50.0,
            help="Higher BLH = better pollutant dispersion"
        )
        
        latitude = st.number_input(
            "Latitude (°N)",
            min_value=-90.0,
            max_value=90.0,
            value=28.47,
            step=0.01,
            help="Geographic latitude. Greater Noida: 28.47°N"
        )
        
        longitude = st.number_input(
            "Longitude (°E)",
            min_value=-180.0,
            max_value=180.0,
            value=77.48,
            step=0.01,
            help="Geographic longitude. Greater Noida: 77.48°E"
        )
        
        # Date input to calculate day of year
        date_input = st.date_input(
            "Date",
            value=datetime.now(),
            help="Select date to calculate day of year"
        )
        day_of_year = date_input.timetuple().tm_yday
    
    st.divider()
    
    # Prediction buttons
    col_btn1, col_btn2 = st.columns(2)
    show_debug = st.checkbox("Show prediction debug info", value=False, key="pred_debug")
    
    with col_btn1:
        predict_btn = st.button("🔮 Predict PM2.5", use_container_width=True, type="primary")
    
    with col_btn2:
        reset_btn = st.button("🔄 Clear Controls", use_container_width=True)
    
    if reset_btn:
        st.rerun()
    
    # Run prediction
    if predict_btn:
        with st.spinner("🔮 Predicting PM2.5 using multi-model ensemble..."):
            try:
                # Build feature vector from current UI values
                feature_vector = np.array([
                    aod,
                    temperature,
                    humidity,
                    wind_speed,
                    boundary_layer_height,
                    latitude,
                    longitude,
                    day_of_year,
                ]).reshape(1, -1)
                
                feature_df = pd.DataFrame(
                    feature_vector,
                    columns=["aod", "temperature", "humidity", "wind_speed", "boundary_layer_height", "lat", "lon", "day_of_year"]
                )
                
                # Load best model (LightGBM) for prediction
                model_dir = config["artifacts"]["models_dir"]
                scenarios = config.get("training_scenarios", [])
                
                # Use full_multi_city_temporal scenario as default
                scenario = "full_multi_city_temporal"
                lgbm_model = load_model(scenario, "lightgbm", model_dir)
                
                if lgbm_model is None:
                    st.error("❌ LightGBM model not found. Please train models first.")
                    return

                # Align inference dataframe to model training feature order
                model_feature_names = getattr(lgbm_model, "feature_names_", None)
                if model_feature_names:
                    missing_features = [feature for feature in model_feature_names if feature not in feature_df.columns]
                    if missing_features:
                        st.error(f"❌ Missing features for prediction: {missing_features}")
                        return
                    feature_df = feature_df[model_feature_names]

                if show_debug:
                    st.write("🔎 Debug: Inference payload sent to model")
                    st.write(feature_df)
                    st.write("🔎 Debug: Model feature order", list(feature_df.columns))
                
                # Predict
                pm25_pred = lgbm_model.predict(feature_df)[0]
                
                # Determine air quality category
                if pm25_pred <= 12:
                    category = "🟢 GOOD"
                    color = "#00ff00"
                    aqi = int(pm25_pred / 12 * 50)
                    advice = "Air quality is satisfactory. Enjoy your outdoor activities!"
                elif pm25_pred <= 35:
                    category = "🟡 MODERATE"
                    color = "#ffff00"
                    aqi = int(50 + (pm25_pred - 12) / 23 * 50)
                    advice = "Sensitive individuals should reduce prolonged outdoor exertion."
                elif pm25_pred <= 55:
                    category = "🟠 UNHEALTHY FOR SENSITIVE GROUPS"
                    color = "#ffa500"
                    aqi = int(100 + (pm25_pred - 35) / 20 * 50)
                    advice = "Members of sensitive groups should avoid outdoor activities."
                elif pm25_pred <= 150:
                    category = "🔴 UNHEALTHY"
                    color = "#ff0000"
                    aqi = int(150 + (pm25_pred - 55) / 95 * 50)
                    advice = "Everyone should avoid outdoor activities."
                else:
                    category = "🟣 HAZARDOUS"
                    color = "#800080"
                    aqi = 500
                    advice = "Everyone should avoid ALL outdoor exertion."
                
                # Display results
                st.success("✅ Prediction successful!")
                
                # Results card
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(
                        "PM2.5 Level",
                        f"{pm25_pred:.2f} µg/m³",
                        delta=f"{pm25_pred - 25:.2f} (vs avg 25)"
                    )
                
                with res_col2:
                    st.metric(
                        "Air Quality Index",
                        aqi,
                        delta="EPA Scale 0-500"
                    )
                
                with res_col3:
                    st.markdown(f"<h3 style='text-align: center; color: {color};'>{category}</h3>", unsafe_allow_html=True)
                
                st.divider()
                
                # Advice
                st.info(f"💡 **Recommendation**: {advice}")
                
                # Input summary
                with st.expander("📋 Input Parameters Summary"):
                    params_df = pd.DataFrame({
                        "Parameter": ["AOD", "Temperature", "Humidity", "Wind Speed", "BLH", "Latitude", "Longitude", "Day of Year"],
                        "Value": [
                            f"{aod:.2f}",
                            f"{temperature:.1f}°C",
                            f"{humidity}%",
                            f"{wind_speed:.1f} m/s",
                            f"{boundary_layer_height:.0f} m",
                            f"{latitude:.4f}°N",
                            f"{longitude:.4f}°E",
                            f"{day_of_year}"
                        ],
                        "Impact on PM2.5": [
                            "Higher AOD = more aerosols",
                            "Affects mixing height & turbulence",
                            "High humidity increases PM2.5",
                            "Higher wind disperses pollution",
                            "Higher BLH better disperses pollution",
                            "Geographic/regional patterns",
                            "Geographic/regional patterns",
                            "Seasonal emissions variation"
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                # Store prediction
                st.session_state.prediction_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pm25": pm25_pred,
                    "aqi": aqi,
                    "category": category,
                    "lat": latitude,
                    "lon": longitude,
                    "temp": temperature,
                    "humidity": humidity,
                    "aod": aod
                })
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                logger.error(f"Prediction error: {e}")
    
    # Prediction history
    if st.session_state.prediction_history:
        st.divider()
        st.markdown("### 📊 **PREDICTION HISTORY**")
        
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Total Predictions", len(hist_df))
        
        with col_stat2:
            avg_pm25 = hist_df['pm25'].mean()
            st.metric("Average PM2.5", f"{avg_pm25:.2f} µg/m³")
        
        with col_stat3:
            max_pm25 = hist_df['pm25'].max()
            st.metric("Peak PM2.5", f"{max_pm25:.2f} µg/m³")
        
        st.divider()
        
        # History table
        st.subheader("Recent Predictions")
        display_cols = ["timestamp", "pm25", "aqi", "category", "lat", "lon"]
        st.dataframe(
            hist_df[display_cols].rename(columns={
                "timestamp": "Time",
                "pm25": "PM2.5 (µg/m³)",
                "aqi": "AQI",
                "category": "Quality",
                "lat": "Latitude",
                "lon": "Longitude"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()


# =====================================================
# MAIN APPLICATION
# =====================================================
def main() -> None:
    """Main Streamlit application."""
    # Apply custom CSS
    inject_custom_css()
    
    # Load configuration
    config = load_config()
    data_loader = load_data_loader()
    
    # Header
    st.markdown(
        '<h1 style="text-align: center; margin-bottom: 10px;">🛰️ AIR POLLUTION COMMAND CENTER</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h3 style="text-align: center; color: var(--accent-cyan); margin-top: -15px;">Multi-City PM2.5 Prediction & Robustness Analytics</h3>',
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    # Initialize session state
    if "analyze_neural" not in st.session_state:
        st.session_state.analyze_neural = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛰️ Command Center",
        "🧠 Neural Analysis",
        "📊 Holographic Comparisons",
        "🌍 Planetary Network",
        "🔮 Prediction Engine",
    ])
    
    with tab1:
        render_command_center(config, data_loader)
    
    with tab2:
        render_neural_analysis(config, data_loader)
    
    with tab3:
        render_holographic_comparisons(config)
    
    with tab4:
        render_planetary_network(config)
    
    with tab5:
        render_prediction_engine(config, data_loader)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--text-dim); margin-top: 20px;">
        <p>🛰️ <b>Air Pollution Multi-City Prediction System</b> | Built with ❤️ + Streamlit + Plotly + Pydeck</p>
        <p>© 2026 | Elite ML Engineering | Non-conventional 3D UI Design</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
