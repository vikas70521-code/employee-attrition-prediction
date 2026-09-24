"""
Employee Attrition Prediction - End-to-End Training Pipeline
============================================================
Steps:
  1. Load & inspect data
  2. Clean data
  3. EDA (plots saved to reports/)
  4. Feature engineering
  5. Preprocessing pipeline (scaling + one-hot encoding)
  6. Compare multiple models with cross-validation
  7. Hyper-parameter tuning of the best model
  8. Decision-threshold tuning (recall-focused, since attrition is imbalanced)
  9. Final evaluation on held-out test set
 10. Save model + metadata for the Flask app

Run:  python src/train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "employee_attrition.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42


# ------------------------------------------------------------------ 1. LOAD
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    print(f"[1] Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"    Attrition rate: {(df['Attrition'] == 'Yes').mean():.1%}")
    return df


# ----------------------------------------------------------------- 2. CLEAN
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print(f"[2] Missing values: {int(df.isna().sum().sum())} | Duplicates: {int(df.duplicated().sum())}")
    df = df.drop_duplicates()
    # Columns with a single value or pure identifiers carry no signal
    constant = [c for c in df.columns if df[c].nunique() == 1]
    drop_cols = list(set(constant + ["EmployeeNumber"]))
    df = df.drop(columns=drop_cols)
    print(f"    Dropped useless columns: {sorted(drop_cols)}")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    return df


# ------------------------------------------------------------------- 3. EDA
def run_eda(df: pd.DataFrame) -> None:
    print("[3] Generating EDA plots -> reports/")

    # Target balance
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="Attrition", data=df, palette=["#4C9F70", "#D64545"])
    ax.set_xticklabels(["Stayed", "Left"])
    ax.set_title("Attrition Class Balance")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(REPORTS / "01_class_balance.png", dpi=150); plt.close()

    # Attrition rate by key categorical columns
    cats = ["OverTime", "JobRole", "MaritalStatus", "BusinessTravel", "Department", "Gender"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, col in zip(axes.ravel(), cats):
        rate = df.groupby(col)["Attrition"].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.values, y=rate.index, ax=ax, palette="rocket")
        ax.set_title(f"Attrition % by {col}"); ax.set_xlabel("%"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(REPORTS / "02_attrition_by_category.png", dpi=150); plt.close()

    # Numeric distributions split by attrition
    nums = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome",
            "TotalWorkingYears", "YearsSinceLastPromotion"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, col in zip(axes.ravel(), nums):
        sns.kdeplot(data=df, x=col, hue="Attrition", fill=True, common_norm=False,
                    palette=["#4C9F70", "#D64545"], ax=ax)
        ax.set_title(f"{col} distribution")
    plt.tight_layout(); plt.savefig(REPORTS / "03_numeric_distributions.png", dpi=150); plt.close()

    # Correlation heatmap
    plt.figure(figsize=(13, 10))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Heatmap"); plt.tight_layout()
    plt.savefig(REPORTS / "04_correlation_heatmap.png", dpi=150); plt.close()


# ------------------------------------------------- 4. FEATURE ENGINEERING
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Also imported by app.py so that training and serving use identical logic."""
    df = df.copy()
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"]
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["AvgYearsPerCompany"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    df["PromotionGap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["OverallSatisfaction"] = df[["EnvironmentSatisfaction", "JobSatisfaction",
                                    "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    return df


# --------------------------------------------------------- 5. PREPROCESSOR
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


# ------------------------------------------------------------------- MAIN
def main():
    df = clean_data(load_data())
    run_eda(df)
    df = add_features(df)
    print("[4] Feature engineering done (5 new features)")

    X, y = df.drop(columns="Attrition"), df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"[5] Train: {len(X_train)} | Test: {len(X_test)}")

    # ---- 6. Compare models (class_weight='balanced' handles imbalance)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring=["roc_auc", "f1", "recall", "precision"])
        rows.append({"Model": name,
                     "ROC-AUC": s["test_roc_auc"].mean(), "F1": s["test_f1"].mean(),
                     "Recall": s["test_recall"].mean(), "Precision": s["test_precision"].mean()})
    comp = pd.DataFrame(rows).set_index("Model").round(3)
    print("\n[6] 5-fold CV comparison:\n", comp, "\n")
    comp.to_csv(REPORTS / "model_comparison.csv")

    plt.figure(figsize=(8, 4.5))
    comp.plot(kind="bar", ax=plt.gca(), colormap="viridis"); plt.xticks(rotation=0)
    plt.title("Model Comparison (5-fold CV)"); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(REPORTS / "05_model_comparison.png", dpi=150); plt.close()

    # ---- 7. Tune the best model by ROC-AUC (logistic regression is usually
    #         strongest on this dataset and is also easy to explain)
    best_name = comp["ROC-AUC"].idxmax()
    print(f"[7] Best model by ROC-AUC: {best_name} -> tuning")
    grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5]},
        "Random Forest": {"clf__max_depth": [5, 8, None], "clf__min_samples_leaf": [1, 3, 5]},
        "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
                              "clf__max_depth": [2, 3]},
    }
    base = Pipeline([("prep", build_preprocessor(X_train)), ("clf", candidates[best_name])])
    search = GridSearchCV(base, grids[best_name], scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"    Best params: {search.best_params_} | CV ROC-AUC: {search.best_score_:.3f}")

    # ---- 8. Threshold tuning using out-of-fold predictions (no test leakage)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    # F2 weights recall 2x more than precision: for HR, missing someone who is
    # about to leave costs more than a needless retention conversation.
    thresholds = np.arange(0.10, 0.90, 0.01)
    scores = [fbeta_score(y_train, oof >= t, beta=2) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    print(f"[8] Optimal decision threshold (max F2): {threshold:.2f}")

    # ---- 9. Final evaluation on the untouched test set
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "model": best_name, "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
    }
    print("\n[9] TEST SET RESULTS")
    print(classification_report(y_test, pred, target_names=["Stayed", "Left"]))
    print(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Left"],
                                            cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1]); axes[1].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[2]); axes[2].set_title("Precision-Recall Curve")
    plt.tight_layout(); plt.savefig(REPORTS / "06_evaluation.png", dpi=150); plt.close()

    # ---- Feature importance / explainability
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = prep.get_feature_names_out()
    imp = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    top = pd.Series(imp, index=[n.split("__", 1)[1] for n in names]).nlargest(15)[::-1]
    plt.figure(figsize=(8, 6)); top.plot(kind="barh", color="#3B6EA8")
    plt.title("Top 15 Drivers of Attrition"); plt.tight_layout()
    plt.savefig(REPORTS / "07_feature_importance.png", dpi=150); plt.close()
    top[::-1].to_csv(REPORTS / "top_features.csv", header=["importance"])

    # ---- 10. Persist artefacts. Defaults let the web form ask for only key fields.
    defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            defaults[col] = str(X[col].mode()[0])
    cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    options = {c: sorted(str(v) for v in X[c].unique()) for c in cat_features}

    joblib.dump(model, MODELS / "attrition_model.joblib")
    (MODELS / "meta.json").write_text(json.dumps(
        {"metrics": metrics, "threshold": threshold, "defaults": defaults,
         "options": options, "columns": X.columns.tolist(),
         "top_features": top[::-1].round(3).to_dict()}, indent=2))
    print("\n[10] Saved model + metadata -> models/  |  plots -> reports/")


if __name__ == "__main__":
    main()
