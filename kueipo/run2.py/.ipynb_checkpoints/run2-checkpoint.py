# -*- coding: utf-8 -*-
"""
RandomForest baseline  +  進階影片層級特徵
-------------------------------------------------
• 影片層級特徵採用 IQR / skew / kurt / ZCR / 頻域能量等 (extract_one)
• 每部影片僅生成 1 行特徵，直接用 RF 做四個任務
• split 仍以 player_id，25% hold-out 估 AUC
• 正類索引由 clf.classes_ 決定；多類 prob 直接 columns 對齊
• 產生 submission.csv (欄位＝sample_submission.csv)
"""

from __future__ import annotations
import argparse, gc, math, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np, pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ---------- 路徑與常數 ----------
PROJ        = Path(__file__).resolve().parent
TR_DIR      = PROJ / "39_Training_Dataset" / "train_data"
TE_DIR      = PROJ / "39_Test_Dataset"     / "test_data"
TR_INFO_CSV = PROJ / "39_Training_Dataset" / "train_info.csv"
TE_INFO_CSV = PROJ / "39_Test_Dataset"     / "test_info.csv"
SAMPLE_SUB  = PROJ / "39_Test_Dataset"     / "sample_submission.csv"

SEG = 25            # 每部影片切 27 段

# ---------- FFT ----------
def _fft(x: np.ndarray) -> np.ndarray:
    """零填充至 2 次冪長度後回傳 rfft 絕對值"""
    n = 1 << (len(x) - 1).bit_length()
    pad = np.zeros(n); pad[:len(x)] = x
    return np.abs(np.fft.rfft(pad))

# ---------- 讀 txt ----------
def _load_txt(p: Path) -> np.ndarray:
    rows = []
    with p.open() as f:
        for idx, ln in enumerate(f):
            cols = ln.strip().split()
            if idx == 0 or len(cols) < 6:       # 跳時間戳
                continue
            rows.append(list(map(int, cols[:6])))
    return np.asarray(rows, dtype=np.int32)

# ---------- 影片層級特徵 ----------
def extract_one(txt: Path, mode: int) -> Dict[str, float]:
    imu = _load_txt(txt)
    seg_idx = np.linspace(0, len(imu), SEG + 1, dtype=int)
    seg_len = np.diff(seg_idx)

    acc = np.linalg.norm(imu[:, :3], axis=1)
    gyr = np.linalg.norm(imu[:, 3:], axis=1)
    acc_fft, gyr_fft = _fft(acc), _fft(gyr)

    seg_acc = [np.linalg.norm(imu[seg_idx[i]:seg_idx[i+1], :3], axis=1).mean()
               if seg_len[i] else 0 for i in range(SEG)]
    seg_acc = np.asarray(seg_acc)

    f: Dict[str, float] = {}

    # 時域統計
    f["acc_mean"]  = seg_acc.mean()
    f["acc_cv"]    = seg_acc.std() / (f["acc_mean"] + 1e-6)
    f["acc_q25"]   = np.percentile(seg_acc, 25)
    f["acc_q75"]   = np.percentile(seg_acc, 75)
    f["acc_skew"]  = float(skew(seg_acc))
    f["acc_kurt"]  = float(kurtosis(seg_acc, fisher=False))
    f["acc_trend"] = np.polyfit(np.arange(SEG), seg_acc, 1)[0]

    jerk = np.diff(acc)
    f["jerk_mean"] = np.abs(jerk).mean()
    f["jerk_std"]  = jerk.std()
    f["jerk_IQR"]  = np.percentile(jerk, 75) - np.percentile(jerk, 25)
    f["acc_ZCR"]   = ((np.sign(acc[1:]) != np.sign(acc[:-1])).sum()) / len(acc)

    f["stroke_len_mean"] = seg_len.mean()
    f["stroke_len_cv"]   = seg_len.std() / (seg_len.mean() + 1e-6)

    diff = np.diff(seg_acc)
    f["seg_diff_mean"] = diff.mean()
    f["seg_diff_std"]  = diff.std()

    # 頻域
    power_a, power_g = acc_fft ** 2, gyr_fft ** 2
    eng_a, eng_g = power_a.mean(), power_g.mean()
    f["energy_acc"] = eng_a
    f["energy_gyr"] = eng_g
    f["energy_ratio"] = eng_a / (eng_g + 1e-6)

    thirds = [int(len(power_a) * 0.33), int(len(power_a) * 0.66)]
    bands = [slice(0, thirds[0]), slice(thirds[0], thirds[1]), slice(thirds[1], None)]
    for i, sli in enumerate(bands, 1):
        f[f"acc_band{i}"] = power_a[sli].sum() / (power_a.sum() + 1e-6)
        f[f"gyr_band{i}"] = power_g[sli].sum() / (power_g.sum() + 1e-6)

    def _spec_stats(power):
        freqs = np.linspace(0, 1, len(power))
        centroid = (freqs * power).sum() / (power.sum() + 1e-6)
        bw = ((freqs - centroid) ** 2 * power).sum() / (power.sum() + 1e-6)
        roll = freqs[np.searchsorted(np.cumsum(power), 0.85 * power.sum())]
        return centroid, bw, roll

    c_a, bw_a, ro_a = _spec_stats(power_a)
    c_g, bw_g, ro_g = _spec_stats(power_g)
    f["centroid_acc"] = c_a; f["centroid_gyr"] = c_g
    f["bandwidth_acc"] = bw_a; f["bandwidth_gyr"] = bw_g
    f["rolloff_acc"] = ro_a; f["rolloff_gyr"] = ro_g

    def _entropy(p):
        p = p / p.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    ent_a, ent_g = _entropy(acc_fft), _entropy(gyr_fft)
    f["fft_entropy_acc"] = ent_a
    f["fft_entropy_gyr"] = ent_g
    f["entropy_diff"] = ent_a - ent_g

    f["energy_norm"] = eng_a / (f["stroke_len_mean"] + 1e-6)
    f["jerk_energy_ratio"] = f["jerk_mean"] / (eng_a + 1e-6)

    # mode one-hot + group
    for m in range(1, 11):
        f[f"mode_{m}"] = int(mode == m)
    f["mode_group"] = 0 if mode in (1, 2) else 1 if mode in (3, 4, 5) \
                      else 2 if mode in (6, 7) else 3
    return f

# ---------- DataFrame 生成 ----------
def build_dataframe() -> tuple[pd.DataFrame, pd.DataFrame]:
    info_tr, info_te = pd.read_csv(TR_INFO_CSV), pd.read_csv(TE_INFO_CSV)

    rows = []
    for txt in TR_DIR.glob("*.txt"):
        uid = int(txt.stem)
        mode = int(info_tr.loc[info_tr.unique_id == uid, "mode"])
        feat = extract_one(txt, mode) | {"unique_id": uid,
                                         "player_id": int(info_tr.loc[info_tr.unique_id == uid, "player_id"])}
        for col in ["gender", "hold racket handed", "play years", "level"]:
            feat[col] = info_tr.loc[info_tr.unique_id == uid, col].values[0]
        rows.append(feat)
    df_tr = pd.DataFrame(rows)

    rows = []
    for txt in TE_DIR.glob("*.txt"):
        uid = int(txt.stem)
        mode = int(info_te.loc[info_te.unique_id == uid, "mode"])
        rows.append(extract_one(txt, mode) | {"unique_id": uid})
    df_te = pd.DataFrame(rows)
    return df_tr, df_te

# ---------- 評估 + submission ----------
def train_and_submit():
    df_tr, df_te = build_dataframe()

    tgt_map = {"gender": "gender",
               "handed": "hold racket handed",
               "years":  "play years",
               "level":  "level"}

    feat_cols = [c for c in df_tr.columns if c not in ["unique_id", "player_id", *tgt_map.values()]]
    scaler = MinMaxScaler().fit(df_tr[feat_cols])
    X_tr = scaler.transform(df_tr[feat_cols])
    X_te = scaler.transform(df_te[feat_cols])

    le = {k: LabelEncoder().fit(df_tr[v]) for k, v in tgt_map.items()}
    y   = {k: le[k].transform(df_tr[v])   for k, v in tgt_map.items()}

    # hold-out split by player
    train_p, test_p = train_test_split(df_tr.player_id.unique(), test_size=0.2, random_state=42)
    mask_val = df_tr.player_id.isin(test_p)
    X_val, y_val = X_tr[mask_val], {k: v[mask_val] for k, v in y.items()}
    X_train, y_train = X_tr[~mask_val], {k: v[~mask_val] for k, v in y.items()}

    def fit_rf(ytr):
        clf = RandomForestClassifier(random_state=42, n_estimators=400,class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, ytr)
        return clf

    clf_g = fit_rf(y_train["gender"])
    clf_h = fit_rf(y_train["handed"])
    clf_y = fit_rf(y_train["years"])
    clf_l = fit_rf(y_train["level"])

    # ---- AUC on hold-out ----
    def auc_binary(clf, yv):
        pos = np.where(clf.classes_ == 1)[0][0]
        return roc_auc_score(yv, clf.predict_proba(X_val)[:, pos])

    def auc_multi(clf, yv):
        return roc_auc_score(yv, clf.predict_proba(X_val), average="micro", multi_class="ovr")

    print("\n── Hold-out AUC ──")
    print(f"gender : {auc_binary(clf_g, y_val['gender']):.4f}")
    print(f"handed : {auc_binary(clf_h, y_val['handed']):.4f}")
    print(f"years  : {auc_multi(clf_y, y_val['years']):.4f}")
    print(f"level  : {auc_multi(clf_l, y_val['level']):.4f}")

    # ---- 使用全量資料重新 fit ----
    clf_g.fit(X_tr, y["gender"])
    clf_h.fit(X_tr, y["handed"])
    clf_y.fit(X_tr, y["years"])
    clf_l.fit(X_tr, y["level"])

    # ---- 預測測試集 ----
    prob_g = clf_g.predict_proba(X_te)
    prob_h = clf_h.predict_proba(X_te)
    prob_y = clf_y.predict_proba(X_te)
    prob_l = clf_l.predict_proba(X_te)

    pos_g = np.where(clf_g.classes_ == 1)[0][0]
    pos_h = np.where(clf_h.classes_ == 1)[0][0]

    sample_cols = list(pd.read_csv(SAMPLE_SUB, nrows=0).columns)
    sub = pd.DataFrame({"unique_id": df_te["unique_id"]})
    sub["gender"] = prob_g[:, pos_g]
    sub["hold racket handed"] = prob_h[:, pos_h]

    # play years / level 多類，欄位需對齊官方
    for i, cls in enumerate(le["years"].classes_):
        sub[f"play years_{i}"] = prob_y[:, i]
    for i, cls in enumerate(le["level"].classes_):
        sub[f"level_{i+2}"] = prob_l[:, i]

    sub = sub[sample_cols].sort_values("unique_id")
    sub.to_csv("submission.csv", index=False, float_format="%.10f")
    print(f"\n📄 submission.csv saved, rows = {len(sub)}")
    gc.collect()

# ---------- main ----------
if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    train_and_submit()
