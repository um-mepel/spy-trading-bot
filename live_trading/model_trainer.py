"""
Model Trainer for Live Trading
==============================
Fetches recent data from Alpaca and trains/updates the LightGBM model.
Runs separately from main.py - designed for live trading use.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, SYMBOL,
    FEATURE_COLUMNS, MIN_TRAINING_SAMPLES
)


class LiveModelTrainer:
    """
    Trains and manages the prediction model for live trading.
    Fetches real-time data from Alpaca and updates model periodically.
    """
    
    def __init__(self, model_dir=None):
        """Initialize the trainer with Alpaca client."""
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.model = None
        self.feature_cols = FEATURE_COLUMNS
        self.training_stats = {}
        
        # Model storage directory
        if model_dir is None:
            model_dir = Path(__file__).parent / "saved_models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / "live_model.pkl"
        self.stats_path = self.model_dir / "training_stats.pkl"
    
    def fetch_minute_data(self, days_back=90):
        """
        Fetch minute-level data from Alpaca for training.
        
        Args:
            days_back: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV minute data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Fetching {SYMBOL} minute data: {start_date.date()} to {end_date.date()}")
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request)
            df = bars.df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                'timestamp': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            if 'symbol' in df.columns:
                df = df[df['symbol'] == SYMBOL]
            
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.sort_values('Datetime').reset_index(drop=True)
            
            print(f"✓ Fetched {len(df):,} minute bars")
            return df
            
        except Exception as e:
            print(f"ERROR fetching data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Calculate technical indicators for the model."""
        df = df.copy()
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_60'] = df['Close'].rolling(window=60, min_periods=1).mean()
        
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Rate of Change
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / (df['Close'].shift(5) + 1e-10)
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10)
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_10'] = df['Returns'].rolling(window=10, min_periods=1).std()
        
        # Price ranges
        df['HL_Range'] = df['High'] - df['Low']
        df['HL_Range_Pct'] = df['HL_Range'] / df['Close']
        df['CO_Range_Pct'] = np.abs(df['Close'] - df['Open']) / df['Close']
        
        return df
    
    def prepare_training_data(self, df, prediction_horizon=20):
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with indicators
            prediction_horizon: How many minutes ahead to predict
            
        Returns:
            X (features), y (target), feature columns
        """
        df = df.copy()
        
        # Target: price change over next N minutes
        df['Target'] = df['Close'].shift(-prediction_horizon) - df['Close']
        
        # Remove last N rows (no future target)
        df = df.iloc[:-prediction_horizon].copy()
        
        # Drop NaN
        df = df.dropna(subset=self.feature_cols + ['Target'])
        
        X = df[self.feature_cols].values
        y = df['Target'].values
        
        return X, y, df
    
    def train_model(self, X, y, n_estimators=200, learning_rate=0.05):
        """
        Train the LightGBM model.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
        """
        print(f"\nTraining model on {len(X):,} samples...")
        
        if HAS_LGB:
            train_set = lgb.Dataset(X, label=y)
            params = {
                'objective': 'regression',
                'metric': 'l2',
                'learning_rate': learning_rate,
                'verbosity': -1,
                'num_leaves': 31,
                'max_depth': -1,
            }
            self.model = lgb.train(params, train_set, num_boost_round=n_estimators)
        else:
            print("LightGBM not available, using sklearn GradientBoosting")
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate
            )
            self.model.fit(X, y)
        
        # Calculate training stats
        if HAS_LGB:
            train_preds = self.model.predict(X)
        else:
            train_preds = self.model.predict(X)
        
        residuals = np.abs(y - train_preds)
        
        self.training_stats = {
            'n_samples': len(X),
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'trained_at': datetime.now().isoformat(),
            'feature_cols': self.feature_cols,
        }
        
        print(f"✓ Model trained successfully")
        print(f"  Mean residual: ${self.training_stats['mean_residual']:.4f}")
        print(f"  Std residual: ${self.training_stats['std_residual']:.4f}")
    
    def save_model(self):
        """Save model and stats to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.stats_path, 'wb') as f:
            pickle.dump(self.training_stats, f)
        
        print(f"✓ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model and stats from disk."""
        if not self.model_path.exists():
            print("No saved model found")
            return False
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if self.stats_path.exists():
            with open(self.stats_path, 'rb') as f:
                self.training_stats = pickle.load(f)
        
        print(f"✓ Model loaded (trained: {self.training_stats.get('trained_at', 'unknown')})")
        return True
    
    def predict(self, features):
        """
        Make a prediction with confidence score.
        
        Args:
            features: Feature vector or DataFrame row
            
        Returns:
            (prediction, confidence) tuple
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if isinstance(features, pd.DataFrame):
            X = features[self.feature_cols].values
        elif isinstance(features, pd.Series):
            X = features[self.feature_cols].values.reshape(1, -1)
        else:
            X = np.array(features).reshape(1, -1)
        
        if HAS_LGB:
            pred = self.model.predict(X)[0]
        else:
            pred = self.model.predict(X)[0]
        
        # Calculate confidence based on prediction magnitude vs training noise
        std = self.training_stats.get('std_residual', 1.0)
        pred_magnitude = abs(pred)
        
        if pred_magnitude < std * 0.5:
            confidence = 0.3
        elif pred_magnitude < std:
            confidence = 0.5
        elif pred_magnitude < std * 2:
            confidence = 0.7
        else:
            confidence = 0.9
        
        return pred, confidence
    
    def run_full_training(self, days_back=90):
        """
        Run the complete training pipeline.
        
        Args:
            days_back: Days of historical data to use
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("LIVE MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Fetch data
        df = self.fetch_minute_data(days_back=days_back)
        if df is None or len(df) < MIN_TRAINING_SAMPLES:
            print(f"ERROR: Insufficient data ({len(df) if df is not None else 0} < {MIN_TRAINING_SAMPLES})")
            return False
        
        # Add indicators
        print("\nCalculating technical indicators...")
        df = self.add_technical_indicators(df)
        
        # Prepare training data
        print("Preparing training data...")
        X, y, processed_df = self.prepare_training_data(df)
        
        if len(X) < MIN_TRAINING_SAMPLES:
            print(f"ERROR: Insufficient samples after processing ({len(X)})")
            return False
        
        # Train model
        self.train_model(X, y)
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return True


def main():
    """Run training pipeline."""
    trainer = LiveModelTrainer()
    success = trainer.run_full_training(days_back=90)
    
    if success:
        print("\n✓ Model ready for live trading")
    else:
        print("\n✗ Training failed")


if __name__ == "__main__":
    main()
