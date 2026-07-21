import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ml.feature_engineering import build_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train(timeframe='4h', ema_fast=20, ema_slow=50, ema_trend=200):
    print("피처 생성 중...")
    df, feature_cols = build_features(timeframe, ema_fast, ema_slow, ema_trend)

    results = {}
    for direction, name in [(1, 'long'), (-1, 'short')]:
        subset = df[df['direction'] == direction].copy()
        if len(subset) < 30:
            continue

        X = subset[feature_cols].values
        y = subset['label'].values

        pos = y.sum()
        neg = len(y) - pos
        scale = neg / pos if pos > 0 else 1

        print(f"\n[{name}] 샘플 {len(X)}개 | 수익 {pos}개({y.mean()*100:.1f}%) | 피처 {len(feature_cols)}개")

        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # XGBoost
        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=scale, eval_metric='logloss',
                             random_state=42, verbosity=0)

        # LightGBM
        lgbm = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=scale, random_state=42, verbose=-1)

        for model_name, model in [('XGB', xgb), ('LGBM', lgbm)]:
            aucs = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y[train_idx])
                pred = model.predict_proba(X_scaled[val_idx])[:, 1]
                aucs.append(roc_auc_score(y[val_idx], pred))
            print(f"  {model_name} 평균 AUC: {np.mean(aucs):.3f} | {[f'{a:.3f}' for a in aucs]}")

        # 더 나은 모델 선택 후 전체 학습
        xgb_auc  = np.mean([roc_auc_score(y[v], xgb.fit(X_scaled[t], y[t]).predict_proba(X_scaled[v])[:,1])
                            for t,v in tscv.split(X_scaled)])
        lgbm_auc = np.mean([roc_auc_score(y[v], lgbm.fit(X_scaled[t], y[t]).predict_proba(X_scaled[v])[:,1])
                            for t,v in tscv.split(X_scaled)])

        best_model = xgb if xgb_auc >= lgbm_auc else lgbm
        best_name  = 'XGB' if xgb_auc >= lgbm_auc else 'LGBM'
        best_auc   = max(xgb_auc, lgbm_auc)
        best_model.fit(X_scaled, y)

        print(f"  → 선택: {best_name} (AUC {best_auc:.3f})")

        joblib.dump(best_model, os.path.join(MODEL_DIR, f"model_{name}_{timeframe}.pkl"))
        joblib.dump(scaler,     os.path.join(MODEL_DIR, f"scaler_{name}_{timeframe}.pkl"))
        joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"features_{timeframe}.pkl"))

        results[name] = best_auc

    print(f"\n학습 완료! Long AUC: {results.get('long', 0):.3f} | Short AUC: {results.get('short', 0):.3f}")
    return results


if __name__ == "__main__":
    train()
