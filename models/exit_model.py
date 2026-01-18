"""
Exit Timing Model - Predicts Short-Term Price Drops
Trained separately to identify when to exit positions.
Complements the Entry Model (LightGBM) for better timing.
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
    from sklearn.ensemble import GradientBoostingClassifier


def _prepare_exit_training(training_data):
    """
    Prepare training data for EXIT model.
    
    Target: Will price drop more than 1% in next 1-2 days?
    This identifies good exit opportunities.
    """
    df = training_data.copy()
    
    # Target: Check if price drops >1% in next 2 days
    # (a small move indicating we should exit to avoid losses)
    df['Next_1d_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Next_2d_Return'] = df['Close'].shift(-2) / df['Close'] - 1
    df['Will_Drop'] = ((df['Next_1d_Return'] < -0.01) | (df['Next_2d_Return'] < -0.02)).astype(int)
    
    # Remove rows where we can't look ahead
    df = df[:-2].copy()
    
    # Exclude price columns from features
    exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                   'Adj Close', 'Next_1d_Return', 'Next_2d_Return', 'Will_Drop', 'index'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure no NaN in features
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['Will_Drop'].values
    
    print(f"Exit Model Training Data:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Will Drop: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Won't Drop: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    
    return X, y, feature_cols


def _prepare_exit_testing(testing_data, feature_cols):
    """
    Prepare testing data for EXIT model.
    """
    df = testing_data.copy()
    
    # Use today's indicators to predict if price will drop tomorrow/next 2 days
    available = [c for c in feature_cols if c in df.columns]
    
    # Shift by 1 day (don't look ahead)
    df[available] = df[available].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    X = df[available].values
    
    print(f"Exit Model Testing Data:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(available)}")
    
    return df, X, available


def train_exit_model(training_data, testing_data, results_dir=None, n_estimators=200, learning_rate=0.05):
    """
    Train a classification model to predict short-term price drops (exits).
    
    Target: Predict if price will drop >1% in next 1-2 days
    This identifies good times to exit positions.
    """
    print("\n" + "="*70)
    print("EXIT TIMING MODEL (Short-Term Drop Prediction)")
    print("="*70 + "\n")

    X_train, y_train, feature_cols = _prepare_exit_training(training_data)
    print(f"  Training features shape: {X_train.shape}\n")

    if HAS_LGB:
        train_set = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary',  # Classification: Will drop or not
            'metric': 'auc',
            'learning_rate': learning_rate,
            'verbosity': -1
        }
        model = lgb.train(params, train_set, num_boost_round=n_estimators)
    else:
        print("lightgbm not available — falling back to sklearn GradientBoostingClassifier")
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train)

    print("✓ Exit Model Training Complete!")

    # Prepare testing set
    test_df, X_test, available = _prepare_exit_testing(testing_data, feature_cols)

    # Predict probability of drop
    if HAS_LGB:
        drop_probs = model.predict(X_test)  # Probability of drop
    else:
        drop_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (drop)

    # Build results DataFrame
    results = test_df[['Date', 'Close']].copy()
    results = results.rename(columns={'Close': 'Actual_Price'})
    results['Drop_Probability'] = drop_probs
    results['Will_Drop'] = (drop_probs > 0.5).astype(int)  # Binary prediction
    results['Exit_Signal_Strength'] = drop_probs  # Probability [0, 1]
    
    # Calculate next day return to evaluate model
    results['Next_1d_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
    results['Actual_Will_Drop'] = (results['Next_1d_Return'] < -0.01).astype(int)
    results['Prediction_Correct'] = (results['Will_Drop'] == results['Actual_Will_Drop']).astype(int)

    print(f"\n✓ Predictions Made: {len(results)} days")
    print(f"  Predicted Drops: {results['Will_Drop'].sum()} ({results['Will_Drop'].mean()*100:.1f}%)")
    print(f"  Predicted No-Drop: {(1-results['Will_Drop']).sum()} ({(1-results['Will_Drop']).mean()*100:.1f}%)")
    print(f"  Prediction Accuracy: {results['Prediction_Correct'].mean()*100:.1f}%")

    # Save results to CSV if directory provided
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / 'exit_model_predictions.csv'
        results.to_csv(output_file, index=False)
        print(f"  Results saved to {output_file.name}")

    return {'results': results, 'model': model}


def main(training_data, testing_data, results_dir=None):
    return train_exit_model(training_data, testing_data, results_dir=results_dir)
