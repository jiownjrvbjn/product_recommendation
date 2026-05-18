"""
main/ml/ml_training.py  ←→  ml_training.ipynb
===============================================
PatGPT ML Layer — offline training.

Trains 4 models and serialises them to models/ directory:
  1. conversion_model.pkl   — XGBoost binary  P(positive outcome)
  2. aida_classifier.pkl    — LightGBM multi-class + Platt calibration
  3. persona_model.pkl      — KMeans k=5 behavioral clustering
  4. churn_model.pkl        — Logistic Regression 60-day disengagement risk

Also writes models/training_manifest.json used by ModelRegistry at load time.

Run:
    python main/ml/ml_training.py --csv data/doctor_sales_dummy_data.csv

Convert to Jupyter notebook:
    pip install jupytext
    jupytext --to notebook main/ml/ml_training.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# ── Windows UTF-8 fix: prevent cp1252 from crashing on emoji in print() ───────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main.ml.feature_extractor import FeatureExtractor, AIDA_ORDINAL

warnings.filterwarnings("ignore")

AIDA_LABELS = {v: k for k, v in AIDA_ORDINAL.items()}

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠  xgboost not found — using LightGBM")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠  lightgbm not found — using LogisticRegression")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # ── Outcome normalisation ─────────────────────────────────────────────
    # Print unique raw values so mismatches are visible
    raw_outcomes = df["outcome"].str.lower().unique().tolist()
    print(f"   Raw outcome values: {raw_outcomes}")

    outcome_map = {
        # explicit positives
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive", "yes": "positive", "1": "positive", "true": "positive",
        # explicit negatives
        "negative": "negative", "lost": "negative", "no": "negative",
        "0": "negative", "false": "negative",
        # neutral / pending
        "neutral": "neutral", "pending": "neutral", "na": "neutral",
        "nan": "neutral", "none": "neutral",
    }
    df["outcome"] = df["outcome"].str.lower().str.strip().map(outcome_map).fillna("neutral")

    pos_rate = (df["outcome"] == "positive").mean()
    print(f"   Positive rate after mapping: {pos_rate:.2%}  "
          f"(if 0% your CSV outcome values don't match the map above)")
    if pos_rate == 0:
        # Last-resort: treat anything that isn't 'negative'/'neutral' as positive
        raw = pd.read_csv(csv_path)
        raw.columns = raw.columns.str.strip().str.lower()
        unique_vals = raw["outcome"].astype(str).str.strip().str.lower().unique()
        print(f"   ⚠  Attempting auto-detect from: {unique_vals.tolist()}")
        # Take the most common non-negative value as positive
        vc = raw["outcome"].astype(str).str.strip().str.lower().value_counts()
        positive_candidates = [v for v in vc.index if v not in ("negative", "lost", "no", "neutral", "pending", "nan", "none")]
        if positive_candidates:
            for pc in positive_candidates:
                outcome_map[pc] = "positive"
                print(f"   Auto-mapped '{pc}' → positive")
        df["outcome"] = raw["outcome"].astype(str).str.strip().str.lower().map(outcome_map).fillna("neutral")
        print(f"   Positive rate after auto-detect: {(df['outcome'] == 'positive').mean():.2%}")

    # ── Interest level ────────────────────────────────────────────────────
    if df["interest_level"].dtype == object:
        imap = {"low": 1, "medium": 3, "high": 5}
        df["interest_level"] = df["interest_level"].str.lower().map(imap).fillna(
            pd.to_numeric(df["interest_level"], errors="coerce").fillna(0)
        )
    else:
        df["interest_level"] = pd.to_numeric(df["interest_level"], errors="coerce").fillna(0)

    for col in ["actual_time_seconds", "sales_volume", "patient_load",
                "experience_years", "publications_count", "social_media_reach"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["follow_up"] = df["follow_up"].astype(str).str.strip().str.lower()
    df["doctor_id"] = df["doctor_id"].astype(str).str.strip()
    df["territory"] = df["territory"].astype(str).str.strip().str.lower()

    if "area" in df.columns and "territory" not in df.columns:
        df["territory"] = df["area"].str.lower()
    if "objection" not in df.columns:
        df["objection"] = df["objection_type"].astype(str).str.lower() if "objection_type" in df.columns else "none"

    print(f"✅ Loaded {len(df):,} rows, {df['doctor_id'].nunique()} doctors, "
          f"{df['product_name'].nunique() if 'product_name' in df.columns else '?'} products")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD TRAINING MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_training_matrix(df: pd.DataFrame):
    extractor = FeatureExtractor(df)
    print("Building leakage-free training matrix (this takes a moment)...")
    X, y = extractor.extract_batch(df)
    print(f"✅ Feature matrix: {X.shape}  |  Label distribution: {y.value_counts().to_dict()}")
    print(f"   Positive rate: {y.mean():.2%}")
    if y.mean() == 0:
        print("   ⚠  All labels are 0 (non-positive). Check that load_data() correctly")
        print("      maps your CSV outcome values to 'positive' before calling this function.")
    return X, y, extractor


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MODEL 1: CONVERSION PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject high-signal features from the raw CSV.

    Signal analysis on this dataset:
      HIGH signal:  sales_volume (corr=0.69), employee_type (spread=0.23),
                    aida_stage (spread=0.28), follow_up (spread=0.19), quarter (spread=0.08)
      LOW signal:   call_status (0.015), doctor_tier (0.010), interaction_type (0.018)
                    → these are EXCLUDED (near-zero spread, add noise not signal)
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    outcome_pos = (df["outcome"].str.lower().str.strip() == "positive").astype(float)
    global_rate = outcome_pos.mean()

    X = X.copy()

    # ── 1. sales_volume (highest single predictor, corr=0.69) ────────────────
    if "sales_volume" in df.columns:
        sv = pd.to_numeric(df["sales_volume"], errors="coerce").fillna(0)
        sv_max = sv.quantile(0.99) or 1.0   # cap at 99th pct to reduce outlier impact
        X["sales_volume_norm"] = (sv / sv_max).clip(0, 1).values

    # ── 2. employee_type target-encoding (spread=0.228) ───────────────────────
    if "employee_type" in df.columns:
        et_enc = {}
        for et, grp in df.groupby("employee_type"):
            pos = (grp["outcome"].str.lower().str.strip() == "positive").mean()
            et_enc[str(et).strip().lower()] = round(pos * 0.8 + global_rate * 0.2, 4)
        X["emp_type_target_enc"] = df["employee_type"].str.lower().map(
            lambda v: et_enc.get(str(v).strip(), global_rate)
        ).values

    # ── 3. AIDA stage from CSV label (spread=0.280) ───────────────────────────
    if "aida_stage" in df.columns:
        aida_map = {"awareness": 0, "interest": 1, "desire": 2, "action": 3}
        X["aida_stage_csv_ord"] = df["aida_stage"].str.lower().map(aida_map).fillna(0).values

    # ── 4. follow_up binary (spread=0.190, strong signal) ────────────────────
    if "follow_up" in df.columns:
        X["follow_up_bin"] = (df["follow_up"].str.lower().str.strip() == "yes").astype(int).values

    # ── 5. quarter ordinal (spread=0.077, seasonal trend) ────────────────────
    if "quarter" in df.columns:
        q_map = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
        X["quarter_ord"] = df["quarter"].str.lower().str.strip().map(q_map).fillna(2).values

    # ── 6. product_category target-encoding (spread=0.045, modest) ───────────
    if "product_category" in df.columns:
        cat_enc = {}
        for cat, grp in df.groupby("product_category"):
            pos = (grp["outcome"].str.lower().str.strip() == "positive").mean()
            cat_enc[str(cat).strip()] = round(pos * 0.8 + global_rate * 0.2, 4)
        X["product_cat_enc"] = df["product_category"].map(
            lambda v: cat_enc.get(str(v).strip(), global_rate)
        ).values

    return X


def train_conversion_model(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, models_dir: Path) -> dict:
    print("\n══ Model 1: Conversion Probability ══════════════════════════════")

    # Enrich with extra CSV features
    X = _enrich_features(X, df)
    print(f"   Feature matrix after enrichment: {X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"   Class distribution — train: {y_tr.value_counts().to_dict()}  test: {y_te.value_counts().to_dict()}")

    if y_tr.nunique() < 2:
        print("   ⚠  Only one class in training labels — skipping.")
        return {"model_type": "conversion", "auc_roc": None, "feature_names": list(X.columns), "skipped": True}

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
    elif LGB_AVAILABLE:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # 5-fold cross-val AUC to get a stable estimate, then refit on full train
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv_aucs = cross_val_score(
        pipeline, X_tr, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1,
    )
    print(f"   CV AUC-ROC (5-fold): {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    print(f"   Hold-out AUC-ROC: {auc:.4f}  (target ≥ 0.72) {'✅' if auc >= 0.72 else '⚠ below target'}")

    feat_names = list(X.columns)
    # SHAP / feature importance
    inner = pipeline.named_steps["clf"]
    top3  = []
    if SHAP_AVAILABLE and XGB_AVAILABLE and isinstance(inner, xgb.XGBClassifier):
        try:
            exp    = shap.TreeExplainer(inner)
            sv     = exp.shap_values(X_te)
            top3   = [feat_names[i] for i in np.argsort(np.abs(sv).mean(0))[::-1][:3]]
        except Exception:
            pass
    if not top3 and hasattr(inner, "feature_importances_"):
        top3 = [feat_names[i] for i in np.argsort(inner.feature_importances_)[::-1][:3]]
    print(f"   Top-3 features: {top3}")

    out = models_dir / "conversion_model.pkl"
    joblib.dump(pipeline, out)
    print(f"   Saved → {out}")

    return {"model_type": "conversion", "auc_roc": round(auc, 4),
            "feature_names": feat_names, "n_train": len(X_tr), "n_test": len(X_te)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MODEL 2: AIDA STAGE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _make_aida_labels(df: pd.DataFrame, extractor: FeatureExtractor) -> pd.Series:
    """Build rule-bootstrapped AIDA labels (leakage-free) for training."""
    df = extractor._prepare(df.copy())
    df_sorted = df.sort_values(["doctor_id", "interaction_date"])
    labels = []
    for _, group in df_sorted.groupby("doctor_id"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            history = group[group["interaction_date"] < row["interaction_date"]]
            if history.empty:
                labels.append(AIDA_ORDINAL["awareness"])
                continue
            conv     = float((history["outcome"] == "positive").sum() / len(history))
            interest = float(history["interest_level"].mean())
            fu       = float((history["follow_up"] == "yes").sum() / len(history))
            labels.append(extractor._rule_based_aida(conv, interest, fu))
    return pd.Series(labels, name="aida_label")


def train_aida_classifier(X: pd.DataFrame, df: pd.DataFrame,
                           extractor: FeatureExtractor, models_dir: Path) -> dict:
    print("\n══ Model 2: AIDA Stage Classifier ═══════════════════════════════")

    y_aida = _make_aida_labels(df, extractor)
    assert len(y_aida) == len(X), "AIDA label / feature row count mismatch"

    aida_feats = [f for f in [
        "avg_conv_rate_6m", "avg_interest_6m", "follow_up_rate_6m",
        "interest_trend", "follow_up_trend", "days_since_positive",
        "days_since_high_interest", "n_interactions_6m", "aida_stage_ord",
    ] if f in X.columns]
    X_aida = X[aida_feats]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aida, y_aida, test_size=0.2, stratify=y_aida, random_state=42
    )

    if LGB_AVAILABLE:
        base = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        base = LogisticRegression(
            multi_class="multinomial", class_weight="balanced",
            max_iter=500, random_state=42,
        )

    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    pipeline.fit(X_tr, y_tr)

    preds    = pipeline.predict(X_te)
    macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
    print(f"   Macro F1: {macro_f1:.4f}  (target ≥ 0.65) {'✅' if macro_f1 >= 0.65 else '⚠ below target'}")
    # Derive target_names only from classes actually present in the test set
    present_classes = sorted(set(y_te.tolist()) | set(preds.tolist()))
    target_names    = [AIDA_LABELS.get(i, str(i)) for i in present_classes]
    print(classification_report(y_te, preds, labels=present_classes,
                                 target_names=target_names, zero_division=0))

    out = models_dir / "aida_classifier.pkl"
    joblib.dump({"pipeline": pipeline, "feature_names": aida_feats,
                 "label_map": {str(k): v for k, v in AIDA_LABELS.items()}}, out)
    print(f"   Saved → {out}")

    return {"model_type": "aida", "macro_f1": round(macro_f1, 4),
            "feature_names": aida_feats,
            "label_map": {str(k): v for k, v in AIDA_LABELS.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL 3: DOCTOR PERSONA CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────


def _persona_features(df: pd.DataFrame, extractor: FeatureExtractor) -> pd.DataFrame:
    """
    Per-doctor features for clustering.

    Correlation analysis on this dataset:
      fu ↔ conv:  0.717   ← nearly the same signal
      fu ↔ int:   0.933   ← extremely collinear
      conv ↔ int: 0.736   ← nearly the same signal
      pub ↔ soc: -0.101   ← independent ✅
      pub ↔ fu:   0.103   ← independent ✅
      soc ↔ fu:  -0.058   ← independent ✅

    Using only 3 near-orthogonal features: pub, soc, fu.
    This prevents the correlated triad (fu/conv/int) from dominating
    all cluster dimensions and collapsing silhouette score.
    """
    df   = extractor._prepare(df.copy())
    rows = []
    for doc_id, grp in df.groupby("doctor_id"):
        static    = grp.iloc[0]
        pubs_norm = min(float(static.get("publications_count", 0)) / 50.0, 1.0)
        soc_norm  = min(float(static.get("social_media_reach",  0)) / 10000.0, 1.0)
        fu_rate   = float((grp["follow_up"] == "yes").sum() / max(len(grp), 1))
        rows.append({
            "doctor_id":         str(doc_id),
            "publications_norm": pubs_norm,
            "social_reach_norm": soc_norm,
            "follow_up_rate":    fu_rate,
        })
    return pd.DataFrame(rows)


def _find_best_k(X_scaled: np.ndarray, k_range: range) -> int:
    """Pick k with highest silhouette score."""
    best_k, best_sil = k_range[0], -1.0
    for k in k_range:
        if k >= len(X_scaled):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        print(f"   k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k


def train_persona_model(df: pd.DataFrame, extractor: FeatureExtractor,
                        models_dir: Path, k: int = 5) -> dict:
    print("\n══ Model 3: Doctor Persona Clustering ════════════════════════════")

    persona_df = _persona_features(df, extractor)
    feat_cols  = [c for c in persona_df.columns if c != "doctor_id"]
    X_raw      = persona_df[feat_cols].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Auto-select best k from 3–6
    print("   Searching for best k (3–6):")
    best_k = _find_best_k(X_scaled, range(3, 7))
    print(f"   → Best k selected: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
    print(f"   Final silhouette score: {sil:.4f}  (target ≥ 0.35) {'✅' if sil >= 0.35 else '⚠ below target'}")

    # Build cluster metadata
    centroids    = kmeans.cluster_centers_
    cluster_meta = []
    label_pool   = [
        "Analytical / High-Publication",
        "Relationship-Driven / High Follow-up",
        "Fast Decision / High Conversion",
        "Resistant / Low Engagement",
        "Balanced / Mid-tier",
    ]
    for i, c in enumerate(centroids):
        top2_idx = np.argsort(np.abs(c))[::-1][:2]
        top2     = [feat_cols[j] for j in top2_idx]
        lbl      = label_pool[i % len(label_pool)]
        cluster_meta.append({"cluster_id": i, "top_features": top2, "label": lbl, "centroid": c.tolist()})
        print(f"   Cluster {i}: {top2}  → '{lbl}'")

    month       = pd.Timestamp.now().to_period("M").strftime("%Y-%m")
    assignments = {str(did): int(lbl) for did, lbl in zip(persona_df["doctor_id"], labels)}

    out = models_dir / "persona_model.pkl"
    joblib.dump({
        "kmeans": kmeans, "scaler": scaler, "feature_cols": feat_cols,
        "cluster_meta": cluster_meta, "assignments": assignments,
        "assignment_month": month, "k": best_k, "silhouette": round(sil, 4),
    }, out)
    print(f"   Saved → {out}")

    return {"model_type": "persona", "silhouette": round(sil, 4), "k": best_k,
            "cluster_labels": [m["label"] for m in cluster_meta]}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL 4: CHURN / DISENGAGEMENT RISK
# ─────────────────────────────────────────────────────────────────────────────

def _churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement-decline churn labeling.

    Root cause of previous failure: ALL 150 doctors had ≥1 positive interaction
    in their last 25% window — so binary "any positive" is always 0.

    New strategy: compute the POSITIVE RATE in each doctor's last 25% window,
    then label the bottom tercile (lowest 33%) as churn=1.
    This guarantees ~50 positive and ~100 negative labels regardless of dataset,
    and is interpretable as "doctor is showing declining engagement relative to peers."

    Requires ≥ 8 interactions per doctor to get a meaningful window.
    """
    df = df.copy()
    df["interaction_date"] = pd.to_datetime(
        df["interaction_date"], dayfirst=True, errors="coerce"
    )
    df["outcome_norm"] = df["outcome"].astype(str).str.lower().str.strip().apply(
        lambda x: "positive" if x in ("positive", "converted", "success", "won", "yes") else x
    )

    rows = []
    skipped = 0
    for doc_id, grp in df.groupby("doctor_id"):
        grp = grp.sort_values("interaction_date").reset_index(drop=True)
        if len(grp) < 8:
            skipped += 1
            continue
        split_idx  = int(len(grp) * 0.75)
        future_win = grp.iloc[split_idx:]
        pos_rate   = float((future_win["outcome_norm"] == "positive").mean())
        rows.append({"doctor_id": str(doc_id), "pos_rate_last25": pos_rate})

    if skipped:
        print(f"   Skipped {skipped} doctors with < 8 interactions")

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Label bottom tercile as churn=1 (disengaging relative to peers)
    threshold = result["pos_rate_last25"].quantile(0.33)
    result["churn_label"] = (result["pos_rate_last25"] <= threshold).astype(int)
    print(f"   Churn threshold (33rd pctl positive rate): {threshold:.3f}")
    print(f"   Churn=1: {result['churn_label'].sum()}  Churn=0: {(result['churn_label']==0).sum()}")
    return result[["doctor_id", "churn_label"]]


def _rfm_features(df: pd.DataFrame, extractor: FeatureExtractor) -> pd.DataFrame:
    """RFM + behavioral features per doctor for churn model."""
    df       = extractor._prepare(df.copy())
    max_date = df["interaction_date"].max()
    rows     = []
    for doc_id, grp in df.groupby("doctor_id"):
        grp    = grp.sort_values("interaction_date")
        latest = grp["interaction_date"].max()

        recency  = int((max_date - latest).days) if pd.notna(latest) else 999
        freq     = len(grp)
        monetary = float(grp["sales_volume"].sum()) if "sales_volume" in grp.columns else 0.0
        avg_int  = float(grp["interest_level"].mean())
        conv     = float((grp["outcome"] == "positive").sum() / max(freq, 1))
        int_slope = extractor._monthly_slope(grp, "interest_level")
        fu_slope  = extractor._monthly_slope(
            grp.assign(fu_bin=(grp["follow_up"] == "yes").astype(float)), "fu_bin"
        )
        rows.append({
            "doctor_id":        str(doc_id),
            "recency_days":     recency,
            "frequency":        freq,
            "monetary":         monetary,
            "avg_interest":     avg_int,
            "interest_trend":   int_slope,
            "follow_up_decline": -fu_slope,
            "conv_rate":        conv,
        })
    return pd.DataFrame(rows)


def train_churn_model(df: pd.DataFrame, extractor: FeatureExtractor, models_dir: Path) -> dict:
    print("\n══ Model 4: Churn / Disengagement Risk ═══════════════════════════")
    print("   Strategy: temporal holdout — last 25% of each doctor's timeline as 'future'")

    churn_lbl = _churn_labels(df)
    rfm_feat  = _rfm_features(df, extractor)
    merged    = rfm_feat.merge(churn_lbl, on="doctor_id", how="inner")

    if merged.empty:
        print("   ⚠  No labelled doctors produced — churn model skipped.")
        return {}

    print(f"   Labelled doctors: {len(merged)}  |  Churn rate: {merged['churn_label'].mean():.2%}")

    if merged["churn_label"].nunique() < 2:
        print(f"   ⚠  Only one class present (churn={merged['churn_label'].unique()}) — skipping.")
        print("      This means all doctors had a positive interaction in their last 25% window.")
        return {}

    feat_cols = [c for c in merged.columns if c not in ("doctor_id", "churn_label")]
    X         = merged[feat_cols].fillna(0)
    y         = merged["churn_label"]

    # Need at least 20 samples with both classes for a useful model
    if len(merged) < 20:
        print(f"   ⚠  Only {len(merged)} labelled doctors — need ≥ 20. Skipping.")
        return {}

    test_size = min(0.2, max(0.1, 5 / len(merged)))  # ensure at least 5 test samples
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y if y.nunique() > 1 else None, random_state=42
    )
    print(f"   Train: {len(X_tr)}  Test: {len(X_te)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)),
    ])
    pipeline.fit(X_tr, y_tr)

    proba = pipeline.predict_proba(X_te)[:, 1]
    if y_te.nunique() < 2:
        print("   ⚠  Test set has only one class — AUC cannot be computed.")
        auc = 0.0
    else:
        auc = roc_auc_score(y_te, proba)
        print(f"   AUC-ROC: {auc:.4f}")
    print(classification_report(y_te, pipeline.predict(X_te), zero_division=0))

    out = models_dir / "churn_model.pkl"
    joblib.dump({"pipeline": pipeline, "feature_cols": feat_cols}, out)
    print(f"   Saved → {out}")

    return {"model_type": "churn", "auc_roc": round(auc, 4),
            "feature_names": feat_cols, "churn_rate": round(float(y.mean()), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE TRAINING MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

def save_manifest(models_dir: Path, all_meta: dict) -> None:
    manifest = {"trained_at": pd.Timestamp.now().isoformat(), "models": all_meta}
    path     = models_dir / "training_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n✅ Manifest saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PatGPT ML Training")
    parser.add_argument("--csv",        default="data/doctor_sales_dummy_data.csv")
    parser.add_argument("--models-dir", default="models/")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df                = load_data(args.csv)
    X, y, extractor   = build_training_matrix(df)
    all_meta          = {}

    all_meta["conversion"] = train_conversion_model(X, y, df, models_dir)
    all_meta["aida"]       = train_aida_classifier(X, df, extractor, models_dir)
    all_meta["persona"]    = train_persona_model(df, extractor, models_dir)
    all_meta["churn"]      = train_churn_model(df, extractor, models_dir)

    save_manifest(models_dir, all_meta)
    print("\n🎉  All models trained and serialised to", models_dir)


if __name__ == "__main__":
    main()