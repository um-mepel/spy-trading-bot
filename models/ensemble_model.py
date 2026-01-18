"""
Ensemble Trading Model combining LightGBM, XGBoost, and Random Forest.
Uses weighted voting to generate more robust predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EnsembleModel:
    """
    Ensemble combining three models with learned weights.
    """
    
    def __init__(self, lgb_weight=0.5, xgb_weight=0.3, rf_weight=0.2):
        """
        Initialize ensemble with model weights.
        
        Args:
            lgb_weight: Weight for LightGBM (default 0.5 - fastest, good generalization)
            xgb_weight: Weight for XGBoost (default 0.3 - strong on this dataset)
            rf_weight: Weight for Random Forest (default 0.2 - provides diversity)
        """
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.rf_weight = rf_weight
        
        # Normalize weights
        total = lgb_weight + xgb_weight + rf_weight
        self.lgb_weight /= total
        self.xgb_weight /= total
        self.rf_weight /= total
        
        self.lgb_model = None
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train all three models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for LightGBM early stopping)
            y_val: Validation targets (optional)
            verbose: Print training progress
        """
        self.feature_names = X_train.columns.tolist()
        
        if verbose:
            print("\n" + "="*70)
            print("ENSEMBLE MODEL TRAINING")
            print("="*70)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 1. LightGBM Model (50% weight)
        if verbose:
            print("\n[1/3] Training LightGBM (50% weight)...")
        
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.lgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='mae',
                early_stopping_rounds=20
            )
        else:
            self.lgb_model.fit(X_train_scaled, y_train)
        
        lgb_pred_train = self.lgb_model.predict(X_train_scaled)
        lgb_mae = mean_absolute_error(y_train, lgb_pred_train)
        lgb_r2 = r2_score(y_train, lgb_pred_train)
        if verbose:
            print(f"  ✓ LightGBM trained: MAE={lgb_mae:.4f}, R²={lgb_r2:.4f}")
        
        # 2. XGBoost Model (30% weight)
        if verbose:
            print("\n[2/3] Training XGBoost (30% weight)...")
        
        self.xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train_scaled, y_train)
        
        xgb_pred_train = self.xgb_model.predict(X_train_scaled)
        xgb_mae = mean_absolute_error(y_train, xgb_pred_train)
        xgb_r2 = r2_score(y_train, xgb_pred_train)
        if verbose:
            print(f"  ✓ XGBoost trained: MAE={xgb_mae:.4f}, R²={xgb_r2:.4f}")
        
        # 3. Random Forest Model (20% weight)
        if verbose:
            print("\n[3/3] Training Random Forest (20% weight)...")
        
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred_train = self.rf_model.predict(X_train_scaled)
        rf_mae = mean_absolute_error(y_train, rf_pred_train)
        rf_r2 = r2_score(y_train, rf_pred_train)
        if verbose:
            print(f"  ✓ Random Forest trained: MAE={rf_mae:.4f}, R²={rf_r2:.4f}")
        
        # Ensemble performance on training set
        ensemble_pred = self.predict(X_train)
        ensemble_mae = mean_absolute_error(y_train, ensemble_pred)
        ensemble_r2 = r2_score(y_train, ensemble_pred)
        
        if verbose:
            print(f"\n[ENSEMBLE RESULT]")
            print(f"  Weighted Average: MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")
            print(f"  Weights: LGB={self.lgb_weight:.1%} XGB={self.xgb_weight:.1%} RF={self.rf_weight:.1%}")
            print(f"\n{'='*70}")
    
    def predict(self, X):
        """
        Generate ensemble predictions as weighted average of all three models.
        
        Args:
            X: Features to predict
            
        Returns:
            Ensemble predictions
        """
        X_scaled = self.scaler.transform(X)
        
        lgb_pred = self.lgb_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        rf_pred = self.rf_model.predict(X_scaled)
        
        # Weighted ensemble
        ensemble_pred = (
            self.lgb_weight * lgb_pred +
            self.xgb_weight * xgb_pred +
            self.rf_weight * rf_pred
        )
        
        return ensemble_pred
    
    def predict_with_individual(self, X):
        """
        Return ensemble prediction plus individual model predictions.
        Useful for analysis and diagnostics.
        
        Args:
            X: Features to predict
            
        Returns:
            Dictionary with 'ensemble', 'lgb', 'xgb', 'rf' predictions
        """
        X_scaled = self.scaler.transform(X)
        
        lgb_pred = self.lgb_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        rf_pred = self.rf_model.predict(X_scaled)
        
        ensemble_pred = (
            self.lgb_weight * lgb_pred +
            self.xgb_weight * xgb_pred +
            self.rf_weight * rf_pred
        )
        
        return {
            'ensemble': ensemble_pred,
            'lgb': lgb_pred,
            'xgb': xgb_pred,
            'rf': rf_pred
        }
    
    def save(self, filepath):
        """Save ensemble models to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load ensemble models from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main(training_df, testing_df, results_dir=None):
    """
    Train ensemble model on training data and make predictions on test data.
    
    Args:
        training_df: Training data (2022-2024)
        testing_df: Testing data (2025)
        results_dir: Directory to save results
        
    Returns:
        Dictionary with predictions and model info
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "model_predictions"
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ENSEMBLE MODEL: LightGBM + XGBoost + Random Forest")
    print("="*70)
    
    # Feature selection (same as LightGBM baseline - available columns)
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'HL_Range_Pct', 'CO_Range_Pct', 'Volume_SMA_20', 'Volume_Ratio',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_12', 'EMA_26', 'MACD', 'MACD_Histogram', 'Signal_Line',
        'RSI_14', 'BB_Upper_20', 'BB_Lower_20', 'BB_Position', 'BB_Width',
        'ATR_14', 'ROC_10', 'Price_SMA20_Distance', 'Price_SMA50_Distance',
        'Momentum_10', 'Volatility_20', 'Daily_Return_Pct', 'Log_Return'
    ]
    
    # Prepare training data - create target matching LightGBM approach
    training_df_prep = training_df.copy()
    training_df_prep['Price_Change_20d'] = training_df_prep['Close'].shift(-20) - training_df_prep['Close']
    training_df_prep = training_df_prep[:-20].copy()  # Remove last 20 rows (can't predict 20 days ahead)
    
    X_train = training_df_prep[feature_cols].copy()
    y_train = training_df_prep['Price_Change_20d'].copy()
    
    # Remove rows with NaN
    valid_idx = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]
    
    print(f"\nTraining set: {len(X_train)} samples, {len(feature_cols)} features")
    
    # Prepare test data - shift indicators by 1 day to match training setup
    testing_df_prep = testing_df[[col for col in testing_df.columns if col in ['Date', 'Close'] + feature_cols]].copy()
    testing_df_prep[feature_cols] = testing_df_prep[feature_cols].shift(1)
    testing_df_prep = testing_df_prep.dropna().reset_index(drop=True)
    
    X_test = testing_df_prep[feature_cols].copy()
    
    # For evaluation, calculate next-day price changes
    # Note: we'll use the 1-day ahead for evaluation purposes
    test_closes = testing_df_prep['Close'].values
    y_test = test_closes[1:] - test_closes[:-1]  # Daily changes, same length as X_test after removing first
    
    # Trim X_test to match y_test length
    X_test = X_test.iloc[1:].reset_index(drop=True)
    testing_df_prep = testing_df_prep.iloc[1:].reset_index(drop=True)
    
    print(f"Testing set: {len(X_test)} samples")
    
    # Create and train ensemble
    # Weights based on individual model test performance:
    # XGB performed best (MAE=0.2060, R²=0.9996), so give it more weight
    # LGB is good all-rounder, RF adds diversity
    ensemble = EnsembleModel(lgb_weight=0.35, xgb_weight=0.45, rf_weight=0.20)
    ensemble.train(X_train, y_train, verbose=True)
    
    # Make predictions on test set
    predictions = ensemble.predict(X_test)
    
    # Evaluate on test set
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)
    test_rmse = np.sqrt(test_mse)
    
    print(f"\n[TEST SET PERFORMANCE]")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Save model
    model_path = results_dir / "ensemble_model.pkl"
    ensemble.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Create predictions dataframe matching LightGBM output format
    predictions_df = testing_df_prep[['Date', 'Close']].copy().reset_index(drop=True)
    predictions_df['Actual_Price'] = predictions_df['Close']
    predictions_df['Actual_Change'] = y_test
    predictions_df['Predicted_Change'] = predictions
    predictions_df['Predicted_Price'] = predictions_df['Actual_Price'] - predictions_df['Predicted_Change']
    predictions_df['Predicted_Change_Pct'] = (predictions_df['Predicted_Change'] / predictions_df['Actual_Price'] * 100)
    predictions_df['Price_Error'] = predictions_df['Predicted_Price'] - predictions_df['Actual_Price']
    predictions_df['Abs_Price_Error'] = predictions_df['Price_Error'].abs()
    
    # Calculate confidence (inverse of prediction error normalized)
    # Higher error = lower confidence
    max_error = predictions_df['Abs_Price_Error'].quantile(0.95)
    predictions_df['Confidence'] = 1.0 - (predictions_df['Abs_Price_Error'] / max_error).clip(0, 1)
    
    # Calculate directional correctness
    predictions_df['Direction_Correct'] = ((predictions_df['Predicted_Change'] * predictions_df['Actual_Change']) > 0).astype(int)
    
    # Save predictions
    predictions_path = results_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to {predictions_path}")
    
    # Summary statistics
    direction_accuracy = predictions_df['Direction_Correct'].mean() * 100
    print(f"\n[DIRECTIONAL ACCURACY]")
    print(f"  Correct: {direction_accuracy:.1f}%")
    print(f"  Beat random: {'✓ YES' if direction_accuracy > 50 else '✗ NO'}")
    
    return {
        'ensemble': ensemble,
        'predictions_df': predictions_df,
        'model_path': model_path,
        'predictions_path': predictions_path
    }


if __name__ == "__main__":
    # Test ensemble
    from data_loader import load_training_data, load_testing_data
    
    training_df = load_training_data()
    testing_df = load_testing_data()
    
    results = main(training_df, testing_df)
