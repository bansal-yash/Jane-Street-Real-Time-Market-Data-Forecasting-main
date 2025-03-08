import pandas as pd
import time
import threading
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from config import submission_configs


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return torch.clamp(out, -5, 5)


class TimeoutException(Exception):
    """Exception for handling timeout."""
    pass


class MixedSignals:
    def __init__(self, timeout_seconds):
        """Initialize the model and set up processing parameters."""
        self.team_id = submission_configs.get("team_id", -1)
        self.timeout_seconds = timeout_seconds
        self.partial_results = None

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained model checkpoint (Secure Loading)
        checkpoint = torch.load("./LSTM.pth", map_location=self.device, weights_only=True)

        # Extract stored feature set from checkpoint
        self.feature_columns = checkpoint.get("full_feature_set", [])

        # Initialize LSTM model
        input_size = len(self.feature_columns)
        self.model = StockLSTM(input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features_df: pd.DataFrame, target_lags_df: pd.DataFrame):
        """Predict function with timeout handling and robust preprocessing."""

        self.partial_results = pd.DataFrame(index=features_df.index, columns=['row_id', 'target_1_pred', 'target_2_pred'])
        self.partial_results["row_id"] = features_df['row_id']
        self.partial_results[['target_1_pred', 'target_2_pred']] = 0.0  # Default values in case of failure

        def run_prediction():
            """Run model inference with timeout."""
            try:
                # Convert to Polars for fast processing
                features_pl = pl.from_pandas(features_df)
                target_lags_pl = pl.from_pandas(target_lags_df)

                if "date_id" not in features_pl.columns:
                    features_pl = features_pl.with_columns(pl.lit(1).alias("date_id"))

                last_date = features_pl.select(pl.col("date_id").last()).item()

                if "date_id" not in target_lags_pl.columns:
                    target_lags_pl = target_lags_pl.with_columns(pl.lit(last_date-1).alias("date_id"))

                # Ensure target lag column names match training format
                actual_lag_columns = [col for col in target_lags_pl.columns if col.startswith("target_")]
                rename_dict = {col: f"lag_{col}" for col in actual_lag_columns}
                target_lags_pl = target_lags_pl.rename(rename_dict)

                # Ensure date_id aligns correctly for lagged data
                target_lags_pl = target_lags_pl.with_columns(pl.col("date_id") + 1)

                # Join target lag features with main features
                full_data = features_pl.join(target_lags_pl, on=["date_id", "symbol_id"], how="left")

                # Detect categorical columns dynamically
                categorical_features = [col for col in full_data.columns if full_data[col].dtype == pl.Utf8]
                full_data = full_data.to_dummies(columns=categorical_features)

                # Fill missing values
                full_data = full_data.fill_null(0)

                # Ensure all required features exist and in correct order
                missing_features = [col for col in self.feature_columns if col not in full_data.columns]
                if missing_features:
                    for col in missing_features:
                        full_data = full_data.with_columns(pl.lit(0).alias(col))

                # Reorder columns to match training
                full_data = full_data.select(self.feature_columns)

                batch_size = 10000
                num_samples = full_data.shape[0]
                
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_data = full_data.slice(start_idx, end_idx - start_idx)
                    x_input = torch.tensor(batch_data.to_numpy(), dtype=torch.float32).unsqueeze(1).to(self.device)
                    
                    # Perform inference
                    with torch.no_grad():
                        batch_predictions = self.model(x_input).cpu().numpy()
                    
                    # Store predictions in output DataFrame
                    self.partial_results.iloc[start_idx:end_idx, 1:3] = batch_predictions[:, :2].astype(np.float64)

            except Exception as e:
                print(f"Error during prediction: {e}")

        # Start a thread with timeout
        prediction_thread = threading.Thread(target=run_prediction)
        prediction_thread.start()
        prediction_thread.join(timeout=self.timeout_seconds)

        if prediction_thread.is_alive():
            print(f"Warning: Prediction timed out after {self.timeout_seconds} seconds!")
            return self.partial_results  # Return default values

        return self.partial_results


if __name__ == "__main__":
    # Test the implementation

    dummy_tester = MixedSignals(timeout_seconds=10)
    print("Team ID:", dummy_tester.team_id)
    print("Timeout Seconds:", dummy_tester.timeout_seconds)

    # Simulating realistic input data for testing
    dummy_features = pd.DataFrame({
        'row_id': range(10),
        'date_id': [1] * 10,
        'symbol_id': range(10),
        'feature_1': np.random.rand(10),
        'feature_2': np.random.rand(10),
        'feature_09': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'feature_10': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X']
    })

    dummy_target_lags = pd.DataFrame({
        'date_id': [0] * 10,
        'symbol_id': range(10),
        'target_1': np.random.rand(10),
        'target_2': np.random.rand(10),
        'target_3': np.random.rand(10)
    })

    predictions = dummy_tester.predict(dummy_features, dummy_target_lags)
    print(predictions.head())
