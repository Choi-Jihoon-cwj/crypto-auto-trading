import joblib
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')


class MLPredictor:
    def __init__(self, timeframe='4h'):
        self.models  = {}
        self.scalers = {}
        self.feature_cols = joblib.load(os.path.join(MODEL_DIR, f"features_{timeframe}.pkl"))

        for name in ['long', 'short']:
            m_path = os.path.join(MODEL_DIR, f"model_{name}_{timeframe}.pkl")
            s_path = os.path.join(MODEL_DIR, f"scaler_{name}_{timeframe}.pkl")
            if os.path.exists(m_path):
                self.models[name]  = joblib.load(m_path)
                self.scalers[name] = joblib.load(s_path)

    def predict(self, features: dict, direction: str) -> float:
        """신호 수익 확률 반환 (0~1)"""
        if direction not in self.models:
            return 0.5

        import pandas as pd
        X_arr = np.array([[features.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scalers[direction].transform(X_arr)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_cols)
        prob = self.models[direction].predict_proba(X_df)[0][1]
        return prob
