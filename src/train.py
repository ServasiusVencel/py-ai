import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from src.config import (
    DEFAULT_RAW_DATA_PATH,
    RAIN_MODEL_PATH,
    FIRE_MODEL_PATH,
    DROUGHT_MODEL_PATH,
    METRICS_REPORT_PATH,
    RAIN_LABELS,
    FIRE_RISK_LABELS,
    DROUGHT_RISK_LABELS,
)
from src.data_loader import load_aws_dataset
from src.preprocessor import AWSPreprocessor

def train_and_evaluate_model(
    model_name: str,
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels_map: Dict[int, str]
) -> Dict[str, Any]:
    """
    Fits model, computes evaluation metrics, and prints structured evaluation report.
    """
    print(f"\n=======================================================")
    print(f"[*] Training {model_name}...")
    print(f"=======================================================")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    # Compute metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"[+] {model_name} Performance:")
    print(f"    - Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"    - Precision (Macro):{prec_macro:.4f}")
    print(f"    - Recall (Macro):   {rec_macro:.4f}")
    print(f"    - F1-Score (Macro): {f1_macro:.4f}")
    print(f"    - F1-Score (Weight):{f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    present_classes = sorted(list(set(y_test.unique()) | set(np.unique(y_pred))))
    target_names = [labels_map.get(i, str(i)) for i in present_classes]
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, zero_division=0))

    # Feature Importance (Top 10)
    top_features = {}
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(importances)[::-1][:10]
        print("Top 10 Important Features:")
        for idx in sorted_indices:
            feat = feature_names[idx]
            imp = float(importances[idx])
            top_features[feat] = imp
            print(f"    - {feat:<30}: {imp:.4f}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": conf_mat,
        "classification_report": clf_report,
        "top_features": top_features,
    }

def main():
    print("=================================================================")
    print("🚀 IoT AWS Multi-Disaster Prediction System - Model Training 🚀")
    print("=================================================================")

    # 1. Load Raw Dataset
    df_raw = load_aws_dataset(DEFAULT_RAW_DATA_PATH)

    # 2. Preprocess and Feature Engineering
    preprocessor = AWSPreprocessor()
    df_processed = preprocessor.fit_transform(df_raw, save_processed=True)
    feature_cols = preprocessor.feature_columns

    X = df_processed[feature_cols]

    # Time-series / Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df_processed) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    print(f"\n[*] Data Split: Train={len(X_train)} samples, Test={len(X_test)} samples")

    all_metrics = {}

    # -------------------------------------------------------------
    # Skenario 1: Potensi Hujan (Rain Prediction)
    # -------------------------------------------------------------
    y_rain = df_processed["target_rain"]
    y_rain_train, y_rain_test = y_rain.iloc[:split_idx], y_rain.iloc[split_idx:]

    rain_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    metrics_rain = train_and_evaluate_model(
        "Rain Prediction Model",
        rain_clf,
        X_train,
        y_rain_train,
        X_test,
        y_rain_test,
        RAIN_LABELS
    )
    all_metrics["rain_prediction"] = metrics_rain

    # Save Rain Model Bundle
    rain_bundle = {
        "model": rain_clf,
        "features": feature_cols,
        "labels": RAIN_LABELS,
        "type": "rain_classifier",
    }
    joblib.dump(rain_bundle, RAIN_MODEL_PATH)
    print(f"[+] Saved Rain Model to: {RAIN_MODEL_PATH}")

    # -------------------------------------------------------------
    # Skenario 2: Potensi Kebakaran (Fire Risk Prediction)
    # -------------------------------------------------------------
    y_fire = df_processed["target_fire_risk"]
    y_fire_train, y_fire_test = y_fire.iloc[:split_idx], y_fire.iloc[split_idx:]

    fire_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    metrics_fire = train_and_evaluate_model(
        "Fire Risk Prediction Model",
        fire_clf,
        X_train,
        y_fire_train,
        X_test,
        y_fire_test,
        FIRE_RISK_LABELS
    )
    all_metrics["fire_risk_prediction"] = metrics_fire

    # Save Fire Model Bundle
    fire_bundle = {
        "model": fire_clf,
        "features": feature_cols,
        "labels": FIRE_RISK_LABELS,
        "type": "fire_risk_classifier",
    }
    joblib.dump(fire_bundle, FIRE_MODEL_PATH)
    print(f"[+] Saved Fire Risk Model to: {FIRE_MODEL_PATH}")

    # -------------------------------------------------------------
    # Skenario 3: Potensi Kekeringan (Drought Risk Prediction)
    # -------------------------------------------------------------
    y_drought = df_processed["target_drought_risk"]
    y_drought_train, y_drought_test = y_drought.iloc[:split_idx], y_drought.iloc[split_idx:]

    drought_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    metrics_drought = train_and_evaluate_model(
        "Drought Risk Prediction Model",
        drought_clf,
        X_train,
        y_drought_train,
        X_test,
        y_drought_test,
        DROUGHT_RISK_LABELS
    )
    all_metrics["drought_risk_prediction"] = metrics_drought

    # Save Drought Model Bundle
    drought_bundle = {
        "model": drought_clf,
        "features": feature_cols,
        "labels": DROUGHT_RISK_LABELS,
        "type": "drought_risk_classifier",
    }
    joblib.dump(drought_bundle, DROUGHT_MODEL_PATH)
    print(f"[+] Saved Drought Risk Model to: {DROUGHT_MODEL_PATH}")

    # -------------------------------------------------------------
    # Save Metrics Summary Report JSON
    # -------------------------------------------------------------
    with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[+] Saved full training metrics report to: {METRICS_REPORT_PATH}")
    print("=================================================================")
    print("🎉 All 3 AI Models Trained and Serialized Successfully! 🎉")
    print("=================================================================")

if __name__ == "__main__":
    main()
