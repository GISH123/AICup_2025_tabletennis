"""
AI CUP 2025 Spring – Table‑Tennis Smart Racket
================================================
End‑to‑end pipeline with **leak‑free group splits** (player_id),
richer swing‑level feature set (time + frequency domain) and
robust file‑level aggregation.  Produces:

  • local 80/20 player‑wise hold‑out AUCs that track the public LB
  • submission_fixed.csv ready for upload

Author : ChatGPT‑o3  (May 2025)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings, math
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1.  Paths
# ----------------------------------------------------------------------
TRAIN_DIR   = Path('39_Training_Dataset')
TEST_DIR    = Path('39_Test_Dataset')
TRAIN_TXT   = TRAIN_DIR / 'train_data'
TEST_TXT    = TEST_DIR  / 'test_data'

INFO_CSV    = TRAIN_DIR / 'train_info.csv'
TEST_INFO   = TEST_DIR  / 'test_info.csv'
SAMPLE_SUB  = TEST_DIR  / 'sample_submission.csv'

# ----------------------------------------------------------------------
# 2.  Feature engineering helpers
# ----------------------------------------------------------------------
def _rms(x: np.ndarray) -> float:
    return math.sqrt((x ** 2).mean()) if len(x) else 0.0

def _skew(x: np.ndarray, mu: float, sigma: float) -> float:
    if sigma == 0: return 0.0
    return float(((x - mu)**3).mean() / (sigma ** 3))

def _kurtosis(x: np.ndarray, mu: float, sigma: float) -> float:
    if sigma == 0: return 0.0
    return float(((x - mu)**4).mean() / (sigma ** 4))

def _spectral_feats(signal: np.ndarray, sr: int = 200) -> typing.Tuple[float,float,float]:
    """Return mean FFT magnitude, mean PSD and spectral entropy."""
    if len(signal) < 4:
        return 0.0, 0.0, 0.0
    fft = np.fft.rfft(signal - signal.mean())
    mag = np.abs(fft)
    psd = (mag ** 2) / len(signal)
    mag_mean = mag.mean()
    psd_mean = psd.mean()
    p = psd / psd.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / math.log(len(p))
    return float(mag_mean), float(psd_mean), float(ent)

def extract_features(swing: np.ndarray) -> dict:
    # Columns: Ax,Ay,Az,Gx,Gy,Gz (ints)
    Ax, Ay, Az, Gx, Gy, Gz = [swing[:, i].astype(float) for i in range(6)]
    feats = {}
    for name, arr in zip(['Ax','Ay','Az','Gx','Gy','Gz'], [Ax,Ay,Az,Gx,Gy,Gz]):
        mu, sigma = arr.mean(), arr.std()
        feats[f'{name}_mean'] = mu
        feats[f'{name}_std']  = sigma
        feats[f'{name}_rms']  = _rms(arr)
        feats[f'{name}_min']  = arr.min()
        feats[f'{name}_max']  = arr.max()
        feats[f'{name}_skew'] = _skew(arr, mu, sigma)
        feats[f'{name}_kurt'] = _kurtosis(arr, mu, sigma)
    # Magnitude channels
    acc_mag  = np.linalg.norm(swing[:, :3], axis=1)
    gyro_mag = np.linalg.norm(swing[:, 3:], axis=1)
    for lbl, arr in [('acc', acc_mag), ('gyro', gyro_mag)]:
        feats[f'{lbl}_mean'] = arr.mean()
        feats[f'{lbl}_std']  = arr.std()
        feats[f'{lbl}_rms']  = _rms(arr)
        feats[f'{lbl}_min']  = arr.min()
        feats[f'{lbl}_max']  = arr.max()
        feats[f'{lbl}_skew'] = _skew(arr, arr.mean(), arr.std())
        feats[f'{lbl}_kurt'] = _kurtosis(arr, arr.mean(), arr.std())
        mag_mean, psd_mean, ent = _spectral_feats(arr)
        feats[f'{lbl}_fft_mean'] = mag_mean
        feats[f'{lbl}_psd_mean'] = psd_mean
        feats[f'{lbl}_entropy']  = ent
    return feats

# ----------------------------------------------------------------------
# 3.  Dataset builders
# ----------------------------------------------------------------------
def parse_cutpoints(cp_str: str) -> np.ndarray:
    return np.fromstring(cp_str.strip('[]'), sep=' ', dtype=int)

def build_dataset(txt_dir: Path, info_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    rows = []
    for txt_file in tqdm(sorted(txt_dir.glob('*.txt')), desc='Extract swings'):
        fid = int(txt_file.stem)
        meta = info_df.loc[info_df['unique_id'] == fid].iloc[0]
        cps  = parse_cutpoints(meta['cut_point'])
        if len(cps) < 2:
            continue
        data = np.loadtxt(txt_file, skiprows=1)
        for i in range(len(cps) - 1):
            swing = data[cps[i]:cps[i+1]]
            feats = extract_features(swing)
            feats.update(file_id=fid,
                         swing_id=i)
            if is_train:
                feats.update(player_id = meta['player_id'],
                             gender     = str(meta['gender']),
                             handed     = str(meta['hold racket handed']),
                             years      = str(meta['play years']),
                             level      = str(meta['level']))
            rows.append(feats)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 4.  Aggregation rules (file‑level)
# ----------------------------------------------------------------------
def agg_binary(probs: np.ndarray) -> float:
    """probs: shape (S,2) → single probability for positive class."""
    pos = probs[:,1]
    return pos.max() if pos.mean() > 0.5 else pos.min()

def agg_multiclass(probs: np.ndarray) -> np.ndarray:
    """Return prob vector using baseline voting rule."""
    sums = probs.sum(axis=0)
    cls  = sums.argmax()
    best_idx = probs[:, cls].argmax()
    return probs[best_idx]

# ----------------------------------------------------------------------
# 5.  Training one label
# ----------------------------------------------------------------------
def train_label(df_train: pd.DataFrame, label: str, n_splits: int = 5, seed: int = 42):
    multiclass = label in {'years','level'}
    le = LabelEncoder().fit(df_train[label])
    y  = le.transform(df_train[label])
    X  = df_train.drop(columns=['file_id','swing_id','player_id','gender','handed','years','level'])
    groups = df_train['player_id'].values
    params = dict(objective='multiclass' if multiclass else 'binary',
                  metric='multi_logloss' if multiclass else 'auc',
                  num_class=len(le.classes_) if multiclass else 1,
                  learning_rate=0.05,
                  feature_pre_filter=False,
                  seed=seed,
                  verbosity=-1)
    boosters = []
    for tr_idx, va_idx in GroupKFold(n_splits).split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx])
        bst = lgb.train(params, dtr, num_boost_round=1500,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        boosters.append(bst)
    return boosters, le

# ----------------------------------------------------------------------
# 6.  Predict & aggregate to file‑level
# ----------------------------------------------------------------------
def file_level_probs(boosters, le, df: pd.DataFrame, label: str) -> pd.Series:
    multiclass = label in {'years','level'}
    X = df.drop(columns=['file_id','swing_id'])
    # Average booster predictions at swing level
    swing_pred = np.mean([bst.predict(X) for bst in boosters], axis=0)
    file_probs = {}
    for fid, grp in df.groupby('file_id'):
        probs = swing_pred[grp.index]
        if multiclass:
            file_probs[fid] = agg_multiclass(probs)
        else:
            file_probs[fid] = agg_binary(probs)
    return pd.Series(file_probs)

# ----------------------------------------------------------------------
# 7.  MAIN
# ----------------------------------------------------------------------
def main():
    print('▶ Building datasets …')
    info_df  = pd.read_csv(INFO_CSV)
    test_info= pd.read_csv(TEST_INFO)
    df_train = build_dataset(TRAIN_TXT, info_df, is_train=True)
    df_test  = build_dataset(TEST_TXT , test_info, is_train=False)

    # 80/20 player‑wise hold‑out for local evaluation
    train_players, val_players = train_test_split(
        info_df['player_id'].unique(), test_size=0.2, random_state=42)

    train_idx = df_train['player_id'].isin(train_players)
    val_idx   = df_train['player_id'].isin(val_players)

    labels = ['gender','handed','years','level']
    results_val = {}
    submission_parts = []

    for label in labels:
        print(f'—— {label}')
        boosters, le = train_label(df_train[train_idx].copy(), label)
        # Validation
        val_probs = file_level_probs(boosters, le, df_train[val_idx].copy(), label)
        y_val = df_train[val_idx].groupby('file_id')[label].first().loc[val_probs.index]
        if label in {'gender','handed'}:
            y_val_enc = le.transform(y_val)
            auc = roc_auc_score(y_val_enc, val_probs.values)
        else:
            y_val_enc = le.transform(y_val)
            auc = roc_auc_score(pd.get_dummies(y_val_enc),
                                np.vstack(val_probs.values), multi_class='ovr')
        results_val[label] = auc
        print(f'   val AUC = {auc:.4f}')

        # Test set prediction
        test_probs = file_level_probs(boosters, le, df_test.copy(), label)
        submission_parts.append(test_probs)

    # Show local metrics
    print('\nLocal hold‑out AUCs:', results_val,
          '\nMean:', np.mean(list(results_val.values())))

    # ------------------------------------------------------------------
    # Build submission CSV (align rows by unique_id key)
    # ------------------------------------------------------------------
    sub_order = pd.read_csv(SAMPLE_SUB)['unique_id']
    sub_df = pd.DataFrame(index=sub_order)

    # Map each label’s probs into correct columns
    for label, s in zip(labels, submission_parts):
        if label in {'gender','handed'}:
            # binary → one column (prob of positive class)
            col = 'gender' if label=='gender' else 'hold racket handed'
            sub_df[col] = s[sub_df.index].values
        elif label == 'years':
            probs = np.vstack(s[sub_df.index].values)
            sub_df[['play years_0','play years_1','play years_2']] = probs
        else:  # level
            probs = np.vstack(s[sub_df.index].values)
            sub_df[['level_2','level_3','level_4','level_5']] = probs

    sub_df = sub_df.reset_index().rename(columns={'index':'unique_id'})
    fname = f'submission_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    sub_df.to_csv(fname, index=False, float_format='%.10f')
    print('✅ Saved', fname)

if __name__ == '__main__':
    main()
