# Air Pollution Model Comparison Report

## Executive Summary

- **Total Scenarios Evaluated**: 7
- **Total Models Evaluated**: 5
- **Total Evaluations**: 35

## Best Models by Scenario (Robustness Score)

- **all_except_bangalore_to_bangalore**: random_forest (Score: 77.29/100)
- **delhi_standalone**: linear_regression (Score: 38.04/100)
- **delhi_to_mumbai**: linear_regression (Score: 69.63/100)
- **full_multi_city_temporal**: linear_regression (Score: 59.68/100)
- **multi_city_to_mumbai**: random_forest (Score: 66.99/100)
- **mumbai_standalone**: linear_regression (Score: 47.82/100)
- **mumbai_to_delhi**: linear_regression (Score: 40.44/100)

## Detailed Rankings by Scenario

### all_except_bangalore_to_bangalore

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | random_forest | 14.3235 | 11.5828 | 0.7949 | 77.29/100 |
| 2 | linear_regression | 20.0251 | 15.3416 | 0.5990 | 75.94/100 |
| 3 | lightgbm | 11.2318 | 9.2300 | 0.8739 | 63.20/100 |
| 4 | xgboost | 13.4544 | 12.0054 | 0.8190 | 49.14/100 |
| 5 | neural_network | 42.7748 | 31.7132 | -0.8295 | 40.00/100 |

### delhi_standalone

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | linear_regression | 52.6826 | 40.9018 | 0.2872 | 38.04/100 |
| 2 | neural_network | 78.7090 | 67.9602 | -0.5911 | 25.65/100 |
| 3 | xgboost | 53.0821 | 39.5169 | 0.2763 | 16.58/100 |
| 4 | random_forest | 56.4941 | 40.9002 | 0.1803 | 10.82/100 |
| 5 | lightgbm | 57.4263 | 43.7435 | 0.1530 | 9.18/100 |

### delhi_to_mumbai

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | linear_regression | 24.7923 | 20.0318 | 0.4939 | 69.63/100 |
| 2 | lightgbm | 25.8544 | 21.8136 | 0.4496 | 53.37/100 |
| 3 | neural_network | 46.9707 | 38.0928 | -0.8167 | 40.00/100 |
| 4 | random_forest | 39.1993 | 34.6748 | -0.2653 | 11.52/100 |
| 5 | xgboost | 42.9805 | 39.0042 | -0.5211 | 0.00/100 |

### full_multi_city_temporal

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | linear_regression | 34.5027 | 23.5386 | 0.4823 | 59.68/100 |
| 2 | neural_network | 64.6544 | 51.5180 | -0.8180 | 39.95/100 |
| 3 | random_forest | 37.7619 | 25.2572 | 0.3798 | 22.79/100 |
| 4 | lightgbm | 37.7827 | 26.3493 | 0.3792 | 22.75/100 |
| 5 | xgboost | 40.2198 | 28.0097 | 0.2965 | 17.79/100 |

### multi_city_to_mumbai

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | random_forest | 20.4484 | 16.5960 | 0.6557 | 66.99/100 |
| 2 | lightgbm | 15.4367 | 12.3352 | 0.8038 | 65.78/100 |
| 3 | neural_network | 50.3930 | 43.1023 | -1.0910 | 40.00/100 |
| 4 | linear_regression | 35.5718 | 30.2440 | -0.0419 | 34.93/100 |
| 5 | xgboost | 27.7836 | 21.3807 | 0.3644 | 21.86/100 |

### mumbai_standalone

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | linear_regression | 32.9266 | 25.5636 | 0.4501 | 47.82/100 |
| 2 | neural_network | 51.0456 | 41.5377 | -0.3215 | 40.00/100 |
| 3 | xgboost | 34.9270 | 25.8267 | 0.3813 | 22.88/100 |
| 4 | random_forest | 36.1348 | 27.2825 | 0.3378 | 20.27/100 |
| 5 | lightgbm | 36.2749 | 27.3849 | 0.3326 | 19.96/100 |

### mumbai_to_delhi

| Rank | Model | RMSE (Test) | MAE (Test) | R² (Test) | Robustness |
|------|-------|-----------|---------|---------|-------------|
| 1 | linear_regression | 41.4956 | 32.8585 | 0.2391 | 40.44/100 |
| 2 | neural_network | 79.4914 | 67.9220 | -1.7923 | 26.86/100 |
| 3 | lightgbm | 44.4577 | 36.5656 | 0.1266 | 7.60/100 |
| 4 | xgboost | 46.5369 | 38.7757 | 0.0430 | 2.58/100 |
| 5 | random_forest | 47.9582 | 40.0447 | -0.0163 | 0.00/100 |

## Overall Model Performance

| Model | Avg Robustness | Avg Test RMSE | Avg Test R² |
|-------|---------|---------|----------|
| lightgbm | 34.55/100 | 32.6378 | 0.4455 |
| linear_regression | 52.36/100 | 34.5709 | 0.3585 |
| neural_network | 36.07/100 | 59.1484 | -0.8943 |
| random_forest | 29.95/100 | 36.0457 | 0.2953 |
| xgboost | 18.69/100 | 36.9978 | 0.2371 |
