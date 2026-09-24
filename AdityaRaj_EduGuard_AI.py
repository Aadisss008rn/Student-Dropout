"""
EduGuard AI -- Student Burnout & Dropout Early-Warning System
=============================================================
Streamlit application.

Pipeline:
  Data -> Cleaning -> EDA -> Burnout Prediction (OOF stacking)
       -> Dropout Prediction -> Risk Explanation -> Recommended Actions

Architecture:
  Stage 1  Burnout_Level predicted first from raw student features.
  Stage 2  Out-of-fold burnout probabilities (leakage-safe) used as additional
           features for the Dropout_Risk classifier.
  Stage 3  Baseline Dropout model (no burnout features) trained for comparison.
  Inference  Burnout probabilities predicted first, then fed to dropout model.

DISCLAIMER:
  Educational machine-learning prototype only.
  Does NOT provide medical, clinical, or psychological diagnoses.
  All outputs are for informational and academic support purposes only.

Run with:
  streamlit run app.py
"""

# ==============================================================================
# SECTION 1 -- IMPORTS & CONFIGURATION
# ==============================================================================
import sys
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# -- Paths (relative to this file -- works on any machine) --
BASE_DIR    = Path(__file__).parent
DATA_FILE   = BASE_DIR / "student_burnout_dropout_dataset_2.csv"
OUTPUT_DIR  = BASE_DIR / "output"
PLOTS_DIR   = OUTPUT_DIR / "plots"
REPORT_FILE = OUTPUT_DIR / "summary_report.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# -- Reproducibility --
RANDOM_STATE = 42
N_FOLDS      = 5

# -- Schema constants --
BURNOUT_ORDER      = ["Low", "Medium", "High"]
BURNOUT_MAP        = {v: i for i, v in enumerate(BURNOUT_ORDER)}
TARGET_BURNOUT     = "Burnout_Level"
TARGET_DROPOUT     = "Dropout_Risk"
ID_COL             = "Student_ID"
BURNOUT_PROBA_COLS = ["burnout_proba_Low", "burnout_proba_Medium", "burnout_proba_High"]

CATEGORICAL_COLS = [
    "Gender", "Department", "Residence_Type",
    "Part_Time_Job", "Family_Income_Bracket", "Counseling_Access",
]

RISK_TIER_THRESHOLDS = [
    ("CRITICAL", 0.80),
    ("HIGH",     0.60),
    ("MEDIUM",   0.40),
    ("LOW",      0.00),
]

# -- Colors --
C_BLUE   = "#3b82d4"
C_RED    = "#e05c5c"
C_GREEN  = "#34a85a"
C_AMBER  = "#f5a623"

plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "font.family":       "sans-serif",
})


# ==============================================================================
# SECTION 2 -- DATA LOADING & CLEANING
# ==============================================================================
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load CSV, drop Student_ID, impute missing values."""
    if not path.exists():
        st.error(
            f"Dataset not found: `{path.name}`\n\n"
            "Please place `student_burnout_dropout_dataset_2.csv` "
            "in the same directory as `app.py`."
        )
        st.stop()

    df = pd.read_csv(path)
    df = df.drop(columns=[ID_COL], errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df.columns:
        if col in (TARGET_BURNOUT, TARGET_DROPOUT):
            continue
        if df[col].isnull().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def encode_features(df: pd.DataFrame, encoders: dict = None):
    """Label-encode categorical columns. Fit on train, reuse on test/inference."""
    X = df.drop(columns=[TARGET_BURNOUT, TARGET_DROPOUT], errors="ignore").copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        if fit_mode:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            encoders[col] = le
        le       = encoders[col]
        known    = set(le.classes_)
        fallback = le.classes_[0]
        X[col]   = X[col].astype(str).apply(
            lambda v, k=known, fb=fallback: v if v in k else fb
        )
        X[col] = le.transform(X[col])
    return X.astype(float), encoders


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split on Dropout_Risk."""
    y = (df[TARGET_DROPOUT] == "Yes").astype(int)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ==============================================================================
# SECTION 3 -- EDA PLOT GENERATION  (saved to disk, displayed via st.image)
# ==============================================================================
@st.cache_data(show_spinner="Generating EDA visualizations...")
def build_eda_plots(_df: pd.DataFrame):
    """Generate and save all 8 EDA plots. Returns list of Path objects."""

    def _save(fig, name):
        p = PLOTS_DIR / name
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        return p

    paths = []

    # 01 Missing values
    miss = _df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    bars = ax.bar(miss.index, miss.values * 100, color=C_BLUE, edgecolor="white")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Values by Feature (before imputation)")
    ax.set_xticklabels(miss.index, rotation=40, ha="right")
    for bar, val in zip(bars, miss.values * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    paths.append(_save(fig, "01_missing_values.png"))

    # 02 Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    dr = _df[TARGET_DROPOUT].value_counts().reindex(["No", "Yes"])
    axes[0].bar(["No Risk", "At Risk"], dr.values,
                color=[C_GREEN, C_RED], edgecolor="white", width=0.5)
    axes[0].set_title("Dropout Risk Distribution")
    axes[0].set_ylabel("Count")
    for i, val in enumerate(dr.values):
        axes[0].text(i, val + 5, f"{val}  ({val/len(_df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    bl = _df[TARGET_BURNOUT].value_counts().reindex(BURNOUT_ORDER)
    axes[1].bar(BURNOUT_ORDER, bl.values,
                color=[C_GREEN, C_AMBER, C_RED], edgecolor="white", width=0.5)
    axes[1].set_title("Burnout Level Distribution")
    axes[1].set_ylabel("Count")
    for i, val in enumerate(bl.values):
        axes[1].text(i, val + 5, f"{val}  ({val/len(_df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    paths.append(_save(fig, "02_target_distributions.png"))

    # 03 Burnout x Dropout heatmap
    ct = pd.crosstab(_df[TARGET_BURNOUT], _df[TARGET_DROPOUT], normalize="index") * 100
    ct = ct.reindex(BURNOUT_ORDER)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.5, cbar_kws={"label": "% of Burnout group"},
                ax=ax, vmin=0, vmax=100)
    ax.set_title("Burnout Level vs Dropout Risk  (row %)")
    plt.tight_layout()
    paths.append(_save(fig, "03_burnout_vs_dropout_heatmap.png"))

    # 04 Feature distributions
    num_feats = ["Attendance_Percent", "Study_Hours_Per_Day", "Previous_GPA",
                 "Sleep_Hours", "Stress_Level", "Anxiety_Score",
                 "Motivation_Score", "Screen_Time_Hours"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.flat, num_feats):
        for val, color, alpha in [("No", C_BLUE, 0.6), ("Yes", C_RED, 0.5)]:
            data = _df.loc[_df[TARGET_DROPOUT] == val, col].dropna()
            ax.hist(data, bins=20, alpha=alpha, color=color,
                    label=f"Dropout={val}", density=True, edgecolor="none")
        ax.set_title(col.replace("_", " "), fontsize=8)
        ax.set_yticks([])
        ax.legend(fontsize=6)
    plt.suptitle("Feature Distributions by Dropout Risk", fontsize=11)
    plt.tight_layout()
    paths.append(_save(fig, "04_feature_distributions.png"))

    # 05 Correlation matrix
    num_df = _df.select_dtypes(include=np.number)
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 6})
    ax.set_title("Correlation Matrix -- Numerical Features")
    plt.tight_layout()
    paths.append(_save(fig, "05_correlation_matrix.png"))

    # 06 Key feature boxplots
    key_feats = ["Attendance_Percent", "Motivation_Score", "Stress_Level",
                 "Financial_Stress_Score", "Family_Support_Score", "Previous_GPA"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.flat, key_feats):
        data_no  = _df.loc[_df[TARGET_DROPOUT] == "No",  col].dropna()
        data_yes = _df.loc[_df[TARGET_DROPOUT] == "Yes", col].dropna()
        bp = ax.boxplot([data_no, data_yes], labels=["No Risk", "At Risk"],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="gray"),
                        capprops=dict(color="gray"))
        for patch, color in zip(bp["boxes"], [C_GREEN, C_RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(col.replace("_", " "), fontsize=9)
    plt.suptitle("Key Features by Dropout Risk", fontsize=11)
    plt.tight_layout()
    paths.append(_save(fig, "06_boxplots_key_features.png"))

    # 07 Categorical vs Dropout
    cat_feats = ["Gender", "Department", "Part_Time_Job",
                 "Family_Income_Bracket", "Counseling_Access", "Residence_Type"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, cat_feats):
        ct2 = pd.crosstab(_df[col], _df[TARGET_DROPOUT], normalize="index") * 100
        ct2.plot(kind="bar", ax=ax, color=[C_GREEN, C_RED],
                 edgecolor="white", width=0.6)
        ax.set_title(col.replace("_", " ") + " vs Dropout Risk", fontsize=9)
        ax.set_ylabel("% of group")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title="Dropout", fontsize=7)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    paths.append(_save(fig, "07_categorical_vs_dropout.png"))

    # 08 Department burnout heatmap
    dept_bl = pd.crosstab(_df["Department"], _df[TARGET_BURNOUT], normalize="index") * 100
    dept_bl = dept_bl.reindex(columns=BURNOUT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(dept_bl, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.3, ax=ax, cbar_kws={"label": "% of department"})
    ax.set_title("Burnout Level Distribution by Department")
    plt.tight_layout()
    paths.append(_save(fig, "08_dept_burnout_heatmap.png"))

    return paths


# ==============================================================================
# SECTION 4 -- MODEL TRAINING  (cached so Streamlit never retrains on rerun)
# ==============================================================================
def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def _generate_oof_burnout_probas(X_train: np.ndarray,
                                  y_burnout_train: np.ndarray) -> np.ndarray:
    oof = np.zeros((len(X_train), 3))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_burnout_train)):
        clf = _make_rf()
        clf.fit(X_train[tr_idx], y_burnout_train[tr_idx])
        oof[val_idx] = clf.predict_proba(X_train[val_idx])
    return oof


def _eval_dropout(clf, X_test, y_test, label):
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return {
        "label":     label,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred,
                                           target_names=["No", "Yes"]),
    }


@st.cache_resource(show_spinner="Training models (this runs once)...")
def train_all_models(data_path: str):
    """
    Full training pipeline. Returns a bundle dict with all fitted objects.
    Decorated with @st.cache_resource so Streamlit only trains once per session.
    """
    df = load_and_clean(Path(data_path))
    train_df, test_df = split_data(df)

    X_train_df, encoders = encode_features(train_df)
    X_test_df,  _        = encode_features(test_df, encoders=encoders)
    feature_cols_base    = list(X_train_df.columns)

    y_burnout_train = train_df[TARGET_BURNOUT].map(BURNOUT_MAP).values
    y_burnout_test  = test_df[TARGET_BURNOUT].map(BURNOUT_MAP).values
    y_dropout_train = (train_df[TARGET_DROPOUT] == "Yes").astype(int).values
    y_dropout_test  = (test_df[TARGET_DROPOUT]  == "Yes").astype(int).values

    # Burnout model
    scaler_burnout       = StandardScaler()
    X_tr_burn_sc         = scaler_burnout.fit_transform(X_train_df.values)
    X_te_burn_sc         = scaler_burnout.transform(X_test_df.values)
    oof_burnout_probas   = _generate_oof_burnout_probas(X_tr_burn_sc, y_burnout_train)
    burnout_model        = _make_rf()
    burnout_model.fit(X_tr_burn_sc, y_burnout_train)
    test_burnout_probas  = burnout_model.predict_proba(X_te_burn_sc)
    burnout_acc          = accuracy_score(y_burnout_test,
                                          burnout_model.predict(X_te_burn_sc))
    burnout_report       = classification_report(
        y_burnout_test, burnout_model.predict(X_te_burn_sc),
        target_names=BURNOUT_ORDER
    )

    # Dropout models
    X_tr_stacked         = np.hstack([X_train_df.values, oof_burnout_probas])
    X_te_stacked         = np.hstack([X_test_df.values,  test_burnout_probas])
    feature_cols_dropout = feature_cols_base + BURNOUT_PROBA_COLS

    scaler_drop_base     = StandardScaler()
    scaler_drop_stacked  = StandardScaler()
    X_tr_base_sc         = scaler_drop_base.fit_transform(X_train_df.values)
    X_te_base_sc         = scaler_drop_base.transform(X_test_df.values)
    X_tr_stacked_sc      = scaler_drop_stacked.fit_transform(X_tr_stacked)
    X_te_stacked_sc      = scaler_drop_stacked.transform(X_te_stacked)

    dropout_baseline     = _make_rf()
    dropout_baseline.fit(X_tr_base_sc, y_dropout_train)
    dropout_stacked      = _make_rf()
    dropout_stacked.fit(X_tr_stacked_sc, y_dropout_train)

    res_baseline = _eval_dropout(dropout_baseline, X_te_base_sc,
                                 y_dropout_test, "Baseline (no burnout)")
    res_stacked  = _eval_dropout(dropout_stacked,  X_te_stacked_sc,
                                 y_dropout_test, "Stacked (with burnout)")

    # Feature importances
    burnout_fi  = pd.Series(burnout_model.feature_importances_,
                            index=feature_cols_base).sort_values(ascending=False)
    dropout_fi  = pd.Series(dropout_stacked.feature_importances_,
                            index=feature_cols_dropout).sort_values(ascending=False)

    # ROC data
    roc_data = {}
    for res in [res_baseline, res_stacked]:
        fpr, tpr, _ = roc_curve(y_dropout_test, res["y_proba"])
        roc_data[res["label"]] = (fpr, tpr, res["auc"])

    return {
        "df":                    df,
        "encoders":              encoders,
        "scaler_burnout":        scaler_burnout,
        "scaler_drop_stacked":   scaler_drop_stacked,
        "burnout_model":         burnout_model,
        "dropout_stacked":       dropout_stacked,
        "feature_cols_base":     feature_cols_base,
        "feature_cols_dropout":  feature_cols_dropout,
        "burnout_acc":           burnout_acc,
        "burnout_report":        burnout_report,
        "res_baseline":          res_baseline,
        "res_stacked":           res_stacked,
        "burnout_fi":            burnout_fi,
        "dropout_fi":            dropout_fi,
        "roc_data":              roc_data,
        "y_dropout_test":        y_dropout_test,
    }


# ==============================================================================
# SECTION 5 -- STUDENT RISK PROFILER  (inference)
# ==============================================================================
FACTOR_CHECKS = [
    ("Attendance_Percent",     lambda v: v < 70,    "Low attendance (<70%)"),
    ("Motivation_Score",       lambda v: v < 4.0,   "Low motivation score (<4)"),
    ("Stress_Level",           lambda v: v > 7.0,   "High stress level (>7)"),
    ("Anxiety_Score",          lambda v: v > 7.0,   "High anxiety score (>7)"),
    ("Sleep_Hours",            lambda v: v < 5.0,   "Insufficient sleep (<5 hrs/day)"),
    ("Study_Hours_Per_Day",    lambda v: v < 1.5,   "Very low study hours (<1.5 hrs/day)"),
    ("Screen_Time_Hours",      lambda v: v > 8.0,   "High screen time (>8 hrs/day)"),
    ("Financial_Stress_Score", lambda v: v > 7.0,   "High financial stress (>7)"),
    ("Family_Support_Score",   lambda v: v < 3.0,   "Low family support (<3)"),
    ("Previous_GPA",           lambda v: v < 5.0,   "Low previous GPA (<5.0)"),
    ("Backlogs",               lambda v: v >= 2,    "Multiple backlogs (>=2)"),
    ("Exercise_Freq_Per_Week", lambda v: v == 0,    "No regular physical exercise"),
    ("Part_Time_Job",          lambda v: v == "Yes","Holds a part-time job"),
    ("Counseling_Access",      lambda v: v == "No", "No access to counseling"),
]

INTERVENTION_MAP = {
    "Low attendance": [
        "Connect with an academic advisor to discuss attendance barriers.",
        "Review timetable for potential scheduling conflicts.",
    ],
    "Low motivation": [
        "Explore career guidance resources to reconnect with academic goals.",
        "Consider a student mentorship or peer-support program.",
    ],
    "High stress": [
        "Visit the campus wellness centre for stress-management resources.",
        "Try structured study breaks (e.g., Pomodoro method) to reduce overload.",
    ],
    "High anxiety": [
        "Speak with a student well-being officer about available support services.",
        "Explore mindfulness or relaxation workshops offered on campus.",
    ],
    "Insufficient sleep": [
        "Aim for 7-8 hours of sleep; review late-night screen habits.",
        "Speak with student health services if sleep difficulties persist.",
    ],
    "Very low study hours": [
        "Work with a study-skills advisor to build a structured revision plan.",
        "Join a peer study group for accountability and shared learning.",
    ],
    "High screen time": [
        "Set daily screen-time limits using device built-in tools.",
        "Replace one hour of screen time with an outdoor or social activity.",
    ],
    "High financial stress": [
        "Enquire about scholarships, bursaries, or emergency funds at the financial aid office.",
        "Ask a student support officer about flexible payment or fee-deferral options.",
    ],
    "Low family support": [
        "Connect with campus social workers or student community groups.",
        "Explore peer mentoring programs for additional social support.",
    ],
    "Low previous GPA": [
        "Request academic tutoring or supplemental instruction from faculty.",
        "Attend office hours early -- catching up is easier before exams.",
    ],
    "Multiple backlogs": [
        "Meet with your department advisor to plan a manageable catch-up schedule.",
        "Tackle one backlog subject at a time rather than all simultaneously.",
    ],
    "No regular physical exercise": [
        "Even a 20-minute daily walk improves focus and mood significantly.",
        "Try low-commitment campus sport or fitness activities.",
    ],
    "Holds a part-time job": [
        "Ensure part-time work hours do not overlap core study periods.",
        "Check campus job boards for lower-hour, on-campus roles.",
    ],
    "No access to counseling": [
        "Ask your institution about free online or telephone counseling alternatives.",
        "Student unions often provide peer-counseling services at no cost.",
    ],
    "Predicted Burnout Level: HIGH": [
        "High burnout prediction indicates accumulated exhaustion -- rest is productive.",
        "Consider a voluntary course-load reduction with faculty approval.",
    ],
    "Predicted Burnout Level: MEDIUM": [
        "Monitor workload closely; medium burnout can escalate without early action.",
        "Protect personal recovery time -- short, regular breaks matter.",
    ],
}


def get_risk_tier(prob: float) -> str:
    for tier, threshold in RISK_TIER_THRESHOLDS:
        if prob >= threshold:
            return tier
    return "LOW"


def profile_student(student_data, burnout_model, dropout_model,
                    encoders, scaler_burnout, scaler_dropout,
                    feature_cols_burnout, feature_cols_dropout):
    row = pd.DataFrame([student_data]).reindex(columns=feature_cols_burnout)
    for col in CATEGORICAL_COLS:
        if col not in row.columns:
            continue
        le       = encoders[col]
        known    = set(le.classes_)
        fallback = le.classes_[0]
        row[col] = row[col].astype(str).apply(
            lambda v, k=known, fb=fallback: v if v in k else fb
        )
        row[col] = le.transform(row[col])
    row = row.astype(float)

    X_burn     = scaler_burnout.transform(row[feature_cols_burnout].values)
    burn_proba = burnout_model.predict_proba(X_burn)[0]
    burn_pred  = BURNOUT_ORDER[int(np.argmax(burn_proba))]

    aug_row    = row.copy()
    aug_row["burnout_proba_Low"]    = burn_proba[0]
    aug_row["burnout_proba_Medium"] = burn_proba[1]
    aug_row["burnout_proba_High"]   = burn_proba[2]

    X_drop     = scaler_dropout.transform(aug_row[feature_cols_dropout].values)
    drop_proba = float(dropout_model.predict_proba(X_drop)[0][1])
    risk_tier  = get_risk_tier(drop_proba)

    risk_flags = []
    for col, check_fn, label in FACTOR_CHECKS:
        val = student_data.get(col)
        if val is not None:
            try:
                if check_fn(val):
                    risk_flags.append(label)
            except Exception:
                pass

    if burn_pred == "High":
        risk_flags.insert(0, "Predicted Burnout Level: HIGH")
    elif burn_pred == "Medium":
        risk_flags.insert(0, "Predicted Burnout Level: MEDIUM")

    suggestions, seen = [], set()
    for flag in risk_flags[:8]:
        for key, actions in INTERVENTION_MAP.items():
            if key.lower() in flag.lower():
                for action in actions:
                    if action not in seen:
                        seen.add(action)
                        suggestions.append(action)
                break

    return {
        "burnout_level":   burn_pred,
        "burnout_proba":   {k: round(float(v), 3)
                            for k, v in zip(BURNOUT_ORDER, burn_proba)},
        "dropout_prob":    round(drop_proba, 3),
        "risk_tier":       risk_tier,
        "risk_flags":      risk_flags[:8],
        "suggestions":     suggestions[:8],
    }


# ==============================================================================
# SECTION 6 -- STREAMLIT UI HELPERS
# ==============================================================================
TIER_COLOR = {
    "LOW":      "#34a85a",
    "MEDIUM":   "#f5a623",
    "HIGH":     "#e05c5c",
    "CRITICAL": "#7b0000",
}
TIER_EMOJI = {
    "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"
}


def _fi_chart(fi_series: pd.Series, title: str, top_n: int = 15):
    """Horizontal bar chart of feature importances, returned as a Figure."""
    top = fi_series.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [C_RED if i == len(top) - 1 else C_BLUE for i in range(len(top))]
    ax.barh([n.replace("_", " ") for n in top.index], top.values,
            color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _roc_chart(roc_data: dict):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, (fpr, tpr, auc) in roc_data.items():
        color = C_BLUE if "Baseline" in label else C_RED
        ax.plot(fpr, tpr, color=color, label=f"{label}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.9, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Dropout Risk -- ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def _cm_chart(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred No", "Pred Yes"],
                yticklabels=["Actual No", "Actual Yes"],
                linewidths=0.4, cbar=False, annot_kws={"size": 14})
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _model_comparison_chart(res_list: list):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = [r["label"] for r in res_list]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    for ax, metric in zip(axes, metrics):
        vals   = [r[metric] for r in res_list]
        colors = [C_BLUE, C_RED]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(metric.upper().replace("AUC", "ROC-AUC"))
        ax.set_xticklabels(labels, rotation=16, ha="right")
    plt.suptitle("Dropout Risk -- Model Comparison", fontsize=11)
    plt.tight_layout()
    return fig


def _burnout_bar(bp: dict):
    """Horizontal bar chart for burnout probabilities."""
    fig, ax = plt.subplots(figsize=(4.5, 1.8))
    colors  = [C_GREEN, C_AMBER, C_RED]
    vals    = [bp["Low"], bp["Medium"], bp["High"]]
    ax.barh(["Low", "Medium", "High"], vals, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Burnout Level Probabilities")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


# ==============================================================================
# SECTION 7 -- STREAMLIT APP LAYOUT
# ==============================================================================
def main():
    # ---- Page config ----
    st.set_page_config(
        page_title="EduGuard AI",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Header ----
    st.title("🎓 EduGuard AI")
    st.subheader("Student Burnout & Dropout Early-Warning System")
    st.caption(
        "**Educational ML Prototype** — This tool does NOT provide medical, "
        "clinical, or psychological diagnoses. All outputs are for informational "
        "and academic support purposes only."
    )
    st.divider()

    # ---- Sidebar ----
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
        st.markdown("### EduGuard AI")
        st.markdown(
            "An end-to-end early-warning system using a **leakage-safe "
            "two-stage ML pipeline**:\n\n"
            "1. Predict **Burnout Level** (Low / Medium / High)\n"
            "2. Use predicted burnout probabilities to predict **Dropout Risk**"
        )
        st.divider()
        st.markdown("**Dataset:** Synthetic — 800 students, 22 features")
        st.markdown("**Algorithm:** Random Forest (class_weight=balanced)")
        st.markdown("**OOF Folds:** 5-fold stratified CV")
        st.divider()
        st.warning(
            "This prototype was built on a **synthetic** dataset. "
            "Real-world accuracy may differ."
        )

    # ---- Load data & train (cached) ----
    bundle = train_all_models(str(DATA_FILE))
    df     = bundle["df"]

    # ---- EDA plots (cached) ----
    eda_paths = build_eda_plots(df)

    # =========================================================
    # TABS
    # =========================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview",
        "🔍 Exploratory Data Analysis",
        "🤖 Model Performance",
        "👤 Student Risk Profiler",
        "📋 Risk Factors & Methods",
    ])

    # =========================================================
    # TAB 1 -- EXECUTIVE OVERVIEW
    # =========================================================
    with tab1:
        st.header("Executive Overview")

        # KPI metric cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Students", len(df))
        col2.metric("At Dropout Risk",
                    f"{(df[TARGET_DROPOUT]=='Yes').sum()}",
                    f"{(df[TARGET_DROPOUT]=='Yes').mean()*100:.1f}%")
        col3.metric("Burnout: High",
                    f"{(df[TARGET_BURNOUT]=='High').sum()}",
                    f"{(df[TARGET_BURNOUT]=='High').mean()*100:.1f}%")
        col4.metric("Best Dropout AUC",
                    f"{bundle['res_stacked']['auc']:.3f}",
                    f"+{bundle['res_stacked']['auc']-bundle['res_baseline']['auc']:.3f} vs baseline")
        col5.metric("Burnout Model Acc", f"{bundle['burnout_acc']:.3f}")

        st.divider()

        # Key insight callout
        ct = pd.crosstab(df[TARGET_BURNOUT], df[TARGET_DROPOUT], normalize="index") * 100
        high_dropout_pct = ct.loc["High", "Yes"] if "Yes" in ct.columns else 0
        st.info(
            f"**Key Insight:** {high_dropout_pct:.0f}% of students predicted with "
            f"**High Burnout** are also at Dropout Risk — the strongest single "
            f"cross-variable signal in the dataset."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(str(eda_paths[1]), caption="Target Distributions", use_container_width=True)
        with c2:
            st.image(str(eda_paths[2]), caption="Burnout Level vs Dropout Risk", use_container_width=True)

        st.subheader("Dataset Summary")
        miss_df = pd.DataFrame({
            "Column":  df.columns.tolist(),
            "dtype":   [str(df[c].dtype) for c in df.columns],
            "Non-Null": [int(df[c].notnull().sum()) for c in df.columns],
            "Missing":  [int(df[c].isnull().sum()) for c in df.columns],
            "Unique":   [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(miss_df, use_container_width=True, height=420)

    # =========================================================
    # TAB 2 -- EDA
    # =========================================================
    with tab2:
        st.header("Exploratory Data Analysis")

        eda_titles = [
            "01 — Missing Values by Feature",
            "02 — Target Variable Distributions",
            "03 — Burnout Level vs Dropout Risk (heatmap)",
            "04 — Feature Distributions by Dropout Risk",
            "05 — Correlation Matrix",
            "06 — Key Feature Boxplots vs Dropout Risk",
            "07 — Categorical Features vs Dropout Risk",
            "08 — Burnout Distribution by Department",
        ]

        for i, (path, title) in enumerate(zip(eda_paths, eda_titles)):
            st.subheader(title)
            st.image(str(path), use_container_width=True)
            if i < len(eda_paths) - 1:
                st.divider()

    # =========================================================
    # TAB 3 -- MODEL PERFORMANCE
    # =========================================================
    with tab3:
        st.header("Model Performance")

        # -- Burnout model --
        st.subheader("Burnout Level Model  (3-class Random Forest)")
        bc1, bc2 = st.columns([1, 2])
        bc1.metric("Accuracy", f"{bundle['burnout_acc']:.4f}")
        bc1.markdown("**Classes:** Low / Medium / High")
        bc1.markdown("**Class weight:** balanced")
        bc1.markdown("**Architecture:** 5-fold OOF training")
        bc2.code(bundle["burnout_report"], language=None)

        st.divider()

        # -- Dropout model comparison --
        st.subheader("Dropout Risk Models  (Baseline vs Stacked)")
        res_list = [bundle["res_baseline"], bundle["res_stacked"]]

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col, metric, label in [
            (mc1, "accuracy",  "Accuracy"),
            (mc2, "precision", "Precision"),
            (mc3, "recall",    "Recall (Yes)"),
            (mc4, "f1",        "F1 Score"),
            (mc5, "auc",       "ROC-AUC"),
        ]:
            base_val    = bundle["res_baseline"][metric]
            stacked_val = bundle["res_stacked"][metric]
            delta       = stacked_val - base_val
            col.metric(
                label=f"{label}",
                value=f"Stacked: {stacked_val:.3f}",
                delta=f"{delta:+.3f} vs baseline",
            )

        st.pyplot(_model_comparison_chart(res_list), use_container_width=True)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.pyplot(_roc_chart(bundle["roc_data"]), use_container_width=True)
        with dc2:
            st.pyplot(
                _cm_chart(bundle["res_stacked"]["cm"],
                          "Confusion Matrix -- Stacked Model"),
                use_container_width=True,
            )

        st.divider()

        # Detailed classification reports
        with st.expander("Baseline Model -- Classification Report"):
            st.code(bundle["res_baseline"]["report"], language=None)
        with st.expander("Stacked Model -- Classification Report"):
            st.code(bundle["res_stacked"]["report"], language=None)

        st.divider()

        # -- Feature importances --
        st.subheader("Feature Importances")
        fi1, fi2 = st.columns(2)
        with fi1:
            st.pyplot(
                _fi_chart(bundle["burnout_fi"],
                          "Burnout Level -- Top Predictors"),
                use_container_width=True,
            )
        with fi2:
            st.pyplot(
                _fi_chart(bundle["dropout_fi"],
                          "Dropout Risk -- Top Predictors (Stacked)"),
                use_container_width=True,
            )

    # =========================================================
    # TAB 4 -- STUDENT RISK PROFILER
    # =========================================================
    with tab4:
        st.header("Student Risk Profiler")
        st.markdown(
            "Enter a student's details below and click **Predict Risk** to "
            "generate a full risk profile. The system first predicts burnout "
            "level, then uses those probabilities to predict dropout risk."
        )

        with st.form("profiler_form"):
            st.subheader("Academic Information")
            a1, a2, a3, a4 = st.columns(4)
            year_of_study      = a1.selectbox("Year of Study", [1, 2, 3, 4])
            department         = a2.selectbox("Department",
                                              ["Engineering", "Business", "Science",
                                               "Arts", "Medicine", "Law"])
            attendance         = a3.slider("Attendance (%)", 30.0, 100.0, 80.0, 1.0)
            study_hours        = a4.slider("Study Hours / Day", 0.0, 8.0, 3.0, 0.1)

            a5, a6, a7, a8 = st.columns(4)
            gpa                = a5.slider("Previous GPA (0-10)", 2.0, 10.0, 7.0, 0.1)
            backlogs           = a6.number_input("Backlogs", 0, 6, 0, 1)
            part_time_job      = a7.selectbox("Part-Time Job", ["No", "Yes"])
            counseling_access  = a8.selectbox("Counseling Access", ["No", "Yes"])

            st.subheader("Lifestyle & Well-being")
            b1, b2, b3, b4 = st.columns(4)
            sleep_hours        = b1.slider("Sleep Hours / Day", 2.0, 10.0, 6.5, 0.5)
            screen_time        = b2.slider("Screen Time (hrs/day)", 0.5, 12.0, 5.0, 0.5)
            exercise           = b3.slider("Exercise (days/week)", 0, 7, 2, 1)
            social_score       = b4.slider("Social Activity Score (0-10)", 0.0, 10.0, 5.0, 0.5)

            st.subheader("Psychosocial Factors")
            c1, c2, c3, c4 = st.columns(4)
            stress             = c1.slider("Stress Level (0-10)", 0.0, 10.0, 5.0, 0.5)
            anxiety            = c2.slider("Anxiety Score (0-10)", 0.0, 10.0, 5.0, 0.5)
            motivation         = c3.slider("Motivation Score (0-10)", 0.0, 10.0, 6.0, 0.5)
            peer_pressure      = c4.slider("Peer Pressure (0-10)", 0.0, 9.8, 4.5, 0.5)

            st.subheader("Background")
            d1, d2, d3, d4 = st.columns(4)
            age                = d1.number_input("Age", 17, 25, 20, 1)
            gender             = d2.selectbox("Gender", ["Female", "Male", "Other"])
            residence          = d3.selectbox("Residence Type",
                                              ["Hostel", "Day Scholar", "PG/Rented"])
            income_bracket     = d4.selectbox("Family Income Bracket",
                                              ["Low", "Lower-Middle", "Middle",
                                               "Upper-Middle", "High"])

            st.subheader("Financial & Family")
            e1, e2 = st.columns(2)
            fin_stress         = e1.slider("Financial Stress Score (0-10)", 0.0, 10.0, 5.0, 0.5)
            fam_support        = e2.slider("Family Support Score (0-10)", 0.0, 10.0, 6.5, 0.5)

            submitted = st.form_submit_button("🔍  Predict Risk", use_container_width=True,
                                              type="primary")

        if submitted:
            student_data = {
                "Age":                   age,
                "Gender":                gender,
                "Year_of_Study":         year_of_study,
                "Department":            department,
                "Residence_Type":        residence,
                "Attendance_Percent":    attendance,
                "Study_Hours_Per_Day":   study_hours,
                "Previous_GPA":          gpa,
                "Backlogs":              int(backlogs),
                "Sleep_Hours":           sleep_hours,
                "Screen_Time_Hours":     screen_time,
                "Exercise_Freq_Per_Week": exercise,
                "Social_Activity_Score": social_score,
                "Part_Time_Job":         part_time_job,
                "Family_Income_Bracket": income_bracket,
                "Financial_Stress_Score": fin_stress,
                "Family_Support_Score":  fam_support,
                "Stress_Level":          stress,
                "Anxiety_Score":         anxiety,
                "Motivation_Score":      motivation,
                "Peer_Pressure_Score":   peer_pressure,
                "Counseling_Access":     counseling_access,
            }

            profile = profile_student(
                student_data         = student_data,
                burnout_model        = bundle["burnout_model"],
                dropout_model        = bundle["dropout_stacked"],
                encoders             = bundle["encoders"],
                scaler_burnout       = bundle["scaler_burnout"],
                scaler_dropout       = bundle["scaler_drop_stacked"],
                feature_cols_burnout = bundle["feature_cols_base"],
                feature_cols_dropout = bundle["feature_cols_dropout"],
            )

            tier  = profile["risk_tier"]
            color = TIER_COLOR[tier]
            emoji = TIER_EMOJI[tier]

            st.divider()
            st.subheader("Prediction Results")

            # Top metric row
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Burnout Level", profile["burnout_level"])
            r2.metric("Dropout Probability",
                      f"{profile['dropout_prob']*100:.1f}%")
            r3.metric("Risk Tier", f"{emoji} {tier}")

            # Risk tier banner
            st.markdown(
                f"""<div style="background:{color};padding:14px 20px;
                border-radius:8px;color:white;font-size:1.2rem;
                font-weight:bold;text-align:center;margin:12px 0;">
                {emoji} Risk Tier: {tier} &nbsp;|&nbsp;
                Dropout Probability: {profile['dropout_prob']*100:.1f}%
                </div>""",
                unsafe_allow_html=True,
            )

            # Burnout proba chart + dropout gauge
            pb1, pb2 = st.columns([1, 1])
            with pb1:
                st.pyplot(_burnout_bar(profile["burnout_proba"]),
                          use_container_width=True)
            with pb2:
                dp = profile["dropout_prob"]
                fig_g, ax_g = plt.subplots(figsize=(4, 2.2))
                bar = ax_g.barh(["Dropout Risk"], [dp], color=color,
                                edgecolor="white", height=0.4)
                ax_g.barh(["Dropout Risk"], [1 - dp], left=[dp],
                          color="#e5e7eb", edgecolor="white", height=0.4)
                ax_g.set_xlim(0, 1)
                ax_g.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_g.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
                ax_g.set_title(f"Dropout Probability: {dp*100:.1f}%")
                ax_g.axvline(0.40, color="gray", linestyle="--",
                             linewidth=0.8, alpha=0.7)
                ax_g.axvline(0.60, color=C_AMBER, linestyle="--",
                             linewidth=0.8, alpha=0.7)
                ax_g.axvline(0.80, color=C_RED, linestyle="--",
                             linewidth=0.8, alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)

            # Risk flags
            st.subheader("Key Risk Factors")
            if profile["risk_flags"]:
                for flag in profile["risk_flags"]:
                    st.markdown(f"- {flag}")
            else:
                st.success("No significant risk flags detected.")

            # Recommendations
            st.subheader("Recommended Actions")
            st.caption(
                "_Non-clinical academic support suggestions only. "
                "This tool does NOT provide clinical diagnoses._"
            )
            if profile["suggestions"]:
                for i, s in enumerate(profile["suggestions"], 1):
                    st.markdown(f"**{i}.** {s}")
            else:
                st.info("Continue regular monitoring. No actionable flags at this time.")

    # =========================================================
    # TAB 5 -- RISK FACTORS & METHODS
    # =========================================================
    with tab5:
        st.header("Risk Factors & Methodology")

        st.subheader("Top Dropout Risk Factors")
        fi_df = bundle["dropout_fi"].head(10).reset_index()
        fi_df.columns = ["Feature", "Importance Score"]
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df = fi_df[["Rank", "Feature", "Importance Score"]]
        fi_df["Importance Score"] = fi_df["Importance Score"].round(4)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

        st.subheader("Top Burnout Predictors")
        bfi_df = bundle["burnout_fi"].head(10).reset_index()
        bfi_df.columns = ["Feature", "Importance Score"]
        bfi_df["Rank"] = range(1, len(bfi_df) + 1)
        bfi_df = bfi_df[["Rank", "Feature", "Importance Score"]]
        bfi_df["Importance Score"] = bfi_df["Importance Score"].round(4)
        st.dataframe(bfi_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Pipeline Architecture")
        st.code("""
Raw Student Features (22 columns)
        |
        v
+-----------------------------------+
|  Stage 1: Burnout Prediction      |
|  Random Forest (5-fold OOF CV)    |
|  -> Low / Medium / High           |
+---------------+-------------------+
                |
     Out-of-fold burnout probabilities
     [P_Low, P_Medium, P_High]
     (leakage-safe: each row predicted
      by a model never trained on it)
                |
                v
+-----------------------------------+
|  Stage 2: Dropout Prediction      |
|  Stacked Random Forest            |
|  Features = base + burnout probas |
|  -> No / Yes + probability        |
+-----------------------------------+
        """, language=None)

        st.subheader("Risk Tier Thresholds")
        tiers_df = pd.DataFrame({
            "Tier":       ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "Probability Range": ["< 40%", "40% – 60%", "60% – 80%", ">= 80%"],
            "Suggested Response": [
                "Routine monitoring",
                "Proactive check-in",
                "Academic advisor referral",
                "Immediate welfare outreach",
            ],
        })
        st.dataframe(tiers_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Responsible Use")
        st.warning(
            "**Important:** This is an educational machine-learning prototype "
            "built on a synthetic dataset of 800 records. It has NOT been "
            "clinically validated and should NOT be used for autonomous "
            "decisions about student welfare. All outputs require review by "
            "qualified student welfare professionals."
        )
        st.markdown("""
**This tool is designed as an academic support aid, not a disciplinary mechanism.**

Any institutional use must:
- Involve qualified welfare professionals in all decisions
- Respect student privacy and applicable data-protection laws
- Never be used to restrict student academic opportunity
- Be communicated to students with full transparency

**No clinical, medical, or psychological diagnosis is made or implied.**
        """)


# ==============================================================================
if __name__ == "__main__":
    main()
