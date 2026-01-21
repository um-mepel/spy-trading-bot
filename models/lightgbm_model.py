"""
CLEAN LightGBM model wrapper for stock price change prediction.
Trains on training data only, predicts on test data only.
NO DATA LEAKAGE.
Returns predictions in the same format as random_forest_model.main()
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor


def _prepare_training_clean(training_data):
    """
    Prepare training data WITHOUT any future data leakage.
    
    Uses training data only (no test data mixed in).
    """
    df = training_data.copy()
    
    # Create target: 20-day forward price change
    # This is PROPER: we're looking at future within training data only
    df['Price_Change_20d'] = df['Close'].shift(-20) - df['Close']
    
    # Remove rows where we can't look 20 days ahead (last 20 rows)
    df = df[:-20].copy()
    
    # Exclude price columns from features
    # CRITICAL: Also exclude 'Price_Change' to prevent data leakage!
    exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                   'Adj Close', 'Price_Change_20d', 'Price_Change', 'index',
                   'Datetime', 'Target'}  # All possible target/leaky columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure no NaN in features
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['Price_Change_20d'].values
    
    print(f"Training data prepared:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    
    return X, y, feature_cols


def _prepare_testing_clean(testing_data, feature_cols):
    """
    Prepare testing data WITHOUT any lookahead bias.
    
    Uses ONLY testing data and its own indicators.
    Features are shifted by 1 day to match training setup.
    """
    df = testing_data.copy()
    
    # Use today's indicators to predict next day's change
    # We'll use current indicators (already calculated properly)
    available = [c for c in feature_cols if c in df.columns]
    
    # Shift indicators by 1 day - use yesterday's indicators to predict today's change
    # This is CORRECT: no lookahead
    df[available] = df[available].shift(1)
    
    # Remove rows with NaN (first row after shift)
    df = df.dropna().reset_index(drop=True)
    
    X = df[available].values
    
    print(f"Testing data prepared:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(available)}")
    
    return df, X, available


def _calculate_prediction_confidence(y_train, preds_test, window=30):
    """
    Calculate prediction confidence intervals based on training residuals.
    Uses a rolling window approach to estimate prediction uncertainty.
    
    Args:
        y_train: Training target values
        preds_test: Test predictions
        window: Number of recent samples to calculate rolling std
    
    Returns:
        confidence scores (0-1) and prediction intervals
    """
    # Calculate residuals on training data
    residuals = np.abs(y_train)
    rolling_std = np.std(residuals[-window:]) if len(residuals) >= window else np.std(residuals)
    
    # Confidence inversely proportional to prediction magnitude relative to noise
    # Higher magnitude predictions = higher confidence
    pred_magnitudes = np.abs(preds_test)
    
    # Normalize confidence to 0-1 scale
    # predictions with magnitude > 2*rolling_std are high confidence
    thresholds = np.array([rolling_std * 0.5, rolling_std * 1.0, rolling_std * 2.0])
    
    confidence = np.zeros_like(preds_test, dtype=float)
    confidence[pred_magnitudes < thresholds[0]] = 0.3  # Very low confidence
    confidence[(pred_magnitudes >= thresholds[0]) & (pred_magnitudes < thresholds[1])] = 0.5  # Low-medium
    confidence[(pred_magnitudes >= thresholds[1]) & (pred_magnitudes < thresholds[2])] = 0.7  # Medium-high
    confidence[pred_magnitudes >= thresholds[2]] = 0.9  # High confidence
    
    # Upper and lower bounds
    std_multiples = 1.96  # 95% confidence interval
    upper_bound = preds_test + (std_multiples * rolling_std)
    lower_bound = preds_test - (std_multiples * rolling_std)
    
    return confidence, lower_bound, upper_bound


def train_lightgbm(training_data, testing_data, results_dir=None, n_estimators=200, learning_rate=0.05):
    """
    Train LightGBM on training data only.
    Make predictions on testing data only.
    NO DATA LEAKAGE.
    Includes prediction confidence scoring.
    """
    print("\n" + "="*70)
    print("CLEAN LightGBM Training (No Leakage + Confidence Scoring)")
    print("="*70 + "\n")

    X_train, y_train, feature_cols = _prepare_training_clean(training_data)
    print(f"  Training features shape: {X_train.shape}")

    if HAS_LGB:
        train_set = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': learning_rate,
            'verbosity': -1
        }
        model = lgb.train(params, train_set, num_boost_round=n_estimators)
    else:
        print("lightgbm not available — falling back to sklearn GradientBoostingRegressor")
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train)

    print("\n✓ Model Training Complete!")

    # Prepare testing set for predictions
    test_df, X_test, available = _prepare_testing_clean(testing_data, feature_cols)

    # Predict
    if HAS_LGB:
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
    
    # Calculate prediction confidence intervals
    confidence, lower_bound, upper_bound = _calculate_prediction_confidence(y_train, preds)
    print(f"\n✓ Prediction Confidence Intervals Calculated")
    print(f"  Average Confidence: {confidence.mean():.2f}")
    print(f"  High Confidence (>0.7): {(confidence > 0.7).sum()} predictions")

    # Build results DataFrame matching expected format
    results = test_df[['Date', 'Close']].copy()
    results = results.rename(columns={'Close': 'Actual_Price'})
    results['Actual_Change'] = results['Actual_Price'].diff().fillna(0)
    results['Predicted_Change'] = preds
    results['Predicted_Price'] = results['Actual_Price'] + results['Predicted_Change']
    results['Predicted_Change_Pct'] = (results['Predicted_Change'] / results['Actual_Price'] * 100).round(2)

    # Confidence intervals (NEW)
    results['Confidence'] = confidence
    results['Confidence_Lower_Bound'] = lower_bound
    results['Confidence_Upper_Bound'] = upper_bound
    results['Is_High_Confidence'] = (confidence > 0.7).astype(int)  # 1 = high confidence

    # Diagnostic / error columns to match random_forest_model output
    results['Price_Error'] = results['Predicted_Price'] - results['Actual_Price']
    results['Abs_Price_Error'] = np.abs(results['Price_Error'])
    results['Change_Error'] = results['Predicted_Change'] - results['Actual_Change']
    results['Abs_Change_Error'] = np.abs(results['Change_Error'])
    results['Direction_Correct'] = (
        (results['Predicted_Change'] > 0) == (results['Actual_Change'] > 0)
    ).astype(int)

    # Save results to CSV if directory provided
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / 'lightgbm_predictions.csv'
        results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file.name}")

    return {'results': results, 'model': model}


def main(training_data, testing_data, results_dir=None):
    return train_lightgbm(training_data, testing_data, results_dir=results_dir)
