
import ta
import joblib
import numpy as np
import pandas as pd
import mplfinance as mpf
from lightgbm import LGBMClassifier
from func import apply_features, create_targets, SYMBOL

# Phase 4: Feature pruning thresholds
MIN_IMPORTANCE_PCT = 0.05   # Remove features with <5% of max importance
CORRELATION_THRESHOLD = 0.95  # Remove one of pair if correlation > 95%
MAX_FEATURES = 76


def remove_redundant_features(X: pd.DataFrame, importance: np.ndarray, feature_names: list, min_importance_pct: float = MIN_IMPORTANCE_PCT, corr_thresh: float = CORRELATION_THRESHOLD) -> list:
    """
    Phase 1/4: Remove low-importance and highly correlated features.
    - Drop features with importance < 5% of max
    - For correlated pairs (r > 0.95), keep the one with higher importance
    """
    imp_dict = dict(zip(feature_names, importance))
    max_imp = max(importance) if len(importance) > 0 else 1.0

    # Step 1: Remove features with < 5% importance (relative to max)
    kept = [f for f in feature_names if imp_dict.get(f, 0) >= min_importance_pct * max_imp]
    if not kept:
        kept = feature_names  # Fallback if all filtered out

    # Step 2: Remove highly correlated features (keep higher-importance one)
    try:
        X_sub = X[kept].select_dtypes(include=[np.number]).copy()
        if len(X_sub.columns) < 2:
            return kept
        corr_matrix = X_sub.corr().abs()
        to_drop = set()
        for i, col1 in enumerate(corr_matrix.columns):
            if col1 in to_drop:
                continue
            for col2 in corr_matrix.columns[i + 1:]:
                if col2 in to_drop:
                    continue
                if corr_matrix.loc[col1, col2] > corr_thresh:
                    if imp_dict.get(col1, 0) >= imp_dict.get(col2, 0):
                        to_drop.add(col2)
                    else:
                        to_drop.add(col1)
                        break
        final = [f for f in kept if f not in to_drop]
        return final if final else kept
    except Exception:
        return kept







data = pd.read_csv('CSV_FILES/MT5_5M_EURUSD_Exchange_Rate_Dataset.csv') 
df = apply_features(data)
df = create_targets(df)
df.dropna(inplace=True)

all_target = ['T_5M','T_10M','T_15M','T_20M','T_30M']
train_df = df.copy()
X_train = train_df.drop(columns=all_target)

for target in all_target:
    y_train = train_df[target]
    model = LGBMClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X_train.columns.to_list()

    # Phase 1/4: Remove redundant (<5% importance) and correlated features
    filtered_features = remove_redundant_features(X_train, importance, feature_names)
    sort_indx = np.argsort(importance)[::-1]
    by_importance = [feature_names[i] for i in sort_indx]
    top76_features = [f for f in by_importance if f in filtered_features][:MAX_FEATURES]
    if not top76_features:
        top76_features = by_importance[:MAX_FEATURES]

    print(f'TOP {len(top76_features)} FEATURES (after <5% prune + corr removal): ', top76_features[:20], '...' if len(top76_features) > 20 else '')

    X_train_top76 = X_train[top76_features]
    model_top76 = LGBMClassifier(n_estimators=200, random_state=42)
    model_top76.fit(X_train_top76, y_train)

    joblib.dump({"model": model_top76, "features": top76_features}, f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")
    print(f'Model for {target} trained and saved.')
    print('-------------------------------------')