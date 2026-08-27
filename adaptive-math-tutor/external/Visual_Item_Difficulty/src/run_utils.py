"""Staged evaluation shared utilities.

Provides:
  - fixed stratified 5-fold IDs on the 580 train items (difficulty quintile x has_figure)
  - split / fold SHA256 hashing + verification
  - per-item prediction saving (parquet + csv)
  - adapter/head weight saving
  - provenance capture (git commit, config, environment)
"""
import os, json, hashlib, subprocess, datetime
import numpy as np
import pandas as pd

from vdiff_utils import RESULTS_DIR, BASE_DIR

MAIN_DIR = os.path.join(RESULTS_DIR, 'main')              # final three-seed runs
CV_DIR   = os.path.join(RESULTS_DIR, 'model_selection')   # frozen-fold recipe search
FOLDS_JSON   = os.path.join(BASE_DIR, 'data/cv_folds.json')
SEEDS = [17, 42, 2026]   # frozen before the final evaluation


# ---------------------------------------------------------------- hashing ---

def split_sha256(train_df, test_df):
    """SHA256 over sorted train/test QuestionIds — verifies the frozen split."""
    payload = json.dumps({
        'train': sorted(int(q) for q in train_df['QuestionId']),
        'test':  sorted(int(q) for q in test_df['QuestionId']),
    }, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def text_sha256(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True).strip()
    except Exception:
        return 'unknown'


# ------------------------------------------------------------------ folds ---

def verify_split(train_df, test_df, expected_sha=None):
    assert len(train_df) == 580 and len(test_df) == 145, \
        f"split size mismatch: {len(train_df)}/{len(test_df)}"
    inter = set(train_df['QuestionId']) & set(test_df['QuestionId'])
    assert not inter, f"train/test overlap: {sorted(inter)[:5]}"
    sha = split_sha256(train_df, test_df)
    if expected_sha is not None:
        assert sha == expected_sha, f"split sha mismatch: {sha} != {expected_sha}"
    return sha


def load_folds(train_df):
    """Return list of (train_idx, val_idx) into train_df rows, from frozen fold file."""
    meta = json.load(open(FOLDS_JSON))
    verify_split_sha = meta['split_sha256']
    qid_to_pos = {int(q): i for i, q in enumerate(train_df['QuestionId'])}
    folds = []
    for f in meta['folds']:
        val_pos = np.array([qid_to_pos[q] for q in f['val_qids']])
        tr_pos  = np.array(sorted(set(range(len(train_df))) - set(val_pos.tolist())))
        folds.append((tr_pos, val_pos))
    return folds, verify_split_sha


# -------------------------------------------------------------- per-item ----

def per_item_frame(df, preds, *, model_id, route, seed, hp_config_id,
                   final_epoch, input_col, split_sha, extra=None):
    """Build the per-item record for one prediction set."""
    preds = np.asarray(preds, dtype=float)
    gold  = df['difficulty'].to_numpy(dtype=float)
    resid = preds - gold
    out = pd.DataFrame({
        'item_id':        df['QuestionId'].to_numpy(),
        'split':          df['split'].to_numpy(),
        'gold_beta':      gold,
        'pred_beta':      preds,
        'residual':       resid,
        'absolute_error': np.abs(resid),
        'squared_error':  resid ** 2,
        'model_id':       model_id,
        'route':          route,
        'seed':           seed,
        'hp_config_id':   hp_config_id,
        'final_epoch':    final_epoch,
        'has_figure':     (df['visual_description'].fillna('') != '').to_numpy(),
        'q_char_len':     df['text'].astype(str).str.len().to_numpy(),
        'd_char_len':     df['visual_description'].fillna('').astype(str).str.len().to_numpy(),
        'q_sha256':       [text_sha256(t) for t in df['text']],
        'd_sha256':       [text_sha256(t) for t in df['visual_description'].fillna('')],
        'input_sha256':   [text_sha256(t) for t in df[input_col]] if input_col in df else '',
        'split_sha256':   split_sha,
        'git_commit':     git_commit(),
    })
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def save_per_item(frame, out_dir, name='test_predictions'):
    os.makedirs(out_dir, exist_ok=True)
    assert frame['pred_beta'].notna().all() and np.isfinite(frame['pred_beta']).all(), \
        'NaN/Inf in predictions'
    pq = os.path.join(out_dir, f'{name}.parquet')
    cs = os.path.join(out_dir, f'{name}.csv')
    frame.to_parquet(pq, index=False)
    frame.to_csv(cs, index=False)
    print(f"Per-item saved: {pq} ({len(frame)} rows)")


def save_weights(out_dir, adapter_state, head_state):
    """Save LoRA adapter + regression-head state dicts."""
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save(adapter_state, os.path.join(out_dir, 'lora_adapter.pt'))
    torch.save(head_state,    os.path.join(out_dir, 'regression_head.pt'))
    print(f"Weights saved: {out_dir}/lora_adapter.pt + regression_head.pt")


def run_provenance(args_ns, split_sha):
    import torch, transformers, peft
    return {
        'timestamp':    datetime.datetime.now().isoformat(),
        'git_commit':   git_commit(),
        'split_sha256': split_sha,
        'hostname':     os.uname().nodename,
        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'versions': {'torch': torch.__version__,
                     'transformers': transformers.__version__,
                     'peft': peft.__version__},
        'args': {k: v for k, v in vars(args_ns).items()},
    }
