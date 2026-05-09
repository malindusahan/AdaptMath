"""Text LLM + LoRA adapters + regression head, trained end-to-end on Rasch difficulty.

This is the predictor for the question-text (Q) and visual-textualization (Q+D)
interfaces. Two staged modes back the paper's numbers:

  --run_mode cv     one recipe on the frozen 5-fold split of the 580 train items
                    (recipe selection; never touches the test set)
  --run_mode final  retrain on all 580 items for a fixed epoch count and predict
                    the 145 test items; run once per seed in {17, 42, 2026}

Input is selected with --input_mode {text, text_vd}; --vd_json swaps in an
alternative description source (e.g. data/descriptions/Qwen_Qwen2.5-VL-7B-Instruct.json)
to build Q+D. Without --run_mode the script runs a hyperparameter grid against a
single stratified holdout split instead.

Usage:
    CUDA_VISIBLE_DEVICES=0 python llm_lora_regression.py \
        --model_name meta-llama/Llama-3.1-8B --input_mode text_vd \
        --vd_json ../data/descriptions/Qwen_Qwen2.5-VL-7B-Instruct.json \
        --run_mode final --lora_r 16 --lr 1e-4 --epochs 5 --seed 42
"""

import os
# Set HF_HOME in your shell to control where model weights are cached.
# Set TRANSFORMERS_OFFLINE=1 yourself if you run on a node without internet
# and all weights are already in HF_HOME.

import argparse, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(__file__))
from vdiff_utils import load_data, compute_metrics, save_results, RESULTS_DIR
import run_utils


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.enc    = tokenizer(list(texts), truncation=True, padding=True,
                                max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


class RegressionHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.float()).squeeze(-1)


def mean_pool(hidden, attention_mask):
    mask   = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1)


def _loss_fn(preds, labels, loss_type):
    if loss_type == 'huber':
        return F.huber_loss(preds, labels, delta=1.0)
    return F.mse_loss(preds, labels)


def train_eval(model_name, tokenizer, lora_backbone, head,
               tr_texts, tr_labels, val_texts, val_labels,
               lr, batch_size=4, grad_accum=4, max_epochs=10, patience=3,
               loss_type='mse', collect=None):
    """Train with early stopping on val; if val_texts is None, train exactly
    max_epochs on all data with no validation (final-train mode)."""
    device = next(head.parameters()).device
    no_val = val_texts is None

    train_ds = TextDataset(tr_texts, tr_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if not no_val:
        val_ds     = TextDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    trainable_params = list(filter(lambda p: p.requires_grad, lora_backbone.parameters()))
    optimizer  = AdamW(trainable_params + list(head.parameters()), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * max_epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    best_rmse  = float('inf')
    best_bb_state   = None
    best_head_state = None
    best_epoch = 0
    no_improve = 0
    epoch_history = []

    backbone_device = next(lora_backbone.parameters()).device

    for epoch in range(max_epochs):
        lora_backbone.train(); head.train()
        optimizer.zero_grad()
        epoch_loss, n_batches = 0.0, 0
        for step, (enc, labels) in enumerate(train_loader):
            enc    = {k: v.to(backbone_device) for k, v in enc.items()}
            labels = labels.to(device)
            out    = lora_backbone(**enc)
            pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
            preds  = head(pooled)
            loss   = _loss_fn(preds, labels, loss_type) / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum; n_batches += 1
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    trainable_params + list(head.parameters()), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if no_val:
            tr_loss = epoch_loss / max(1, n_batches)
            epoch_history.append({'epoch': epoch + 1, 'train_loss': tr_loss})
            print(f"    epoch {epoch+1}: train loss={tr_loss:.4f}")
            continue

        lora_backbone.eval(); head.eval()
        preds_all, trues_all = [], []
        with torch.no_grad():
            for enc, labels in val_loader:
                enc    = {k: v.to(backbone_device) for k, v in enc.items()}
                out    = lora_backbone(**enc)
                pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
                preds_all.extend(head(pooled).cpu().tolist())
                trues_all.extend(labels.tolist())
        val_rmse = float(np.sqrt(np.mean(
            (np.array(preds_all) - np.array(trues_all)) ** 2)))
        print(f"    epoch {epoch+1}: val RMSE={val_rmse:.4f}")
        epoch_history.append({'epoch': epoch + 1,
                              'train_loss': epoch_loss / max(1, n_batches),
                              'val_rmse': val_rmse})

        if val_rmse < best_rmse - 1e-4:
            best_rmse  = val_rmse
            best_epoch = epoch + 1
            best_bb_state   = {k: v.cpu().clone() for k, v in lora_backbone.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if no_val:
        if collect is not None:
            collect['epoch_history'] = epoch_history
            collect['final_epoch']   = max_epochs
        return None, None

    lora_backbone.load_state_dict(best_bb_state)
    head.load_state_dict(best_head_state)
    lora_backbone.eval(); head.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for enc, labels in val_loader:
            enc    = {k: v.to(backbone_device) for k, v in enc.items()}
            out    = lora_backbone(**enc)
            pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
            preds_all.extend(head(pooled).cpu().tolist())
            trues_all.extend(labels.tolist())
    if collect is not None:
        collect['epoch_history'] = epoch_history
        collect['best_epoch']    = best_epoch
        collect['val_preds']     = preds_all
        collect['val_trues']     = trues_all
    return compute_metrics(trues_all, preds_all), best_rmse


def build_lora_model(model_name, lora_r, lora_targets='attn'):
    backbone = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map='auto')
    hidden_size = backbone.config.hidden_size
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if lora_targets == 'attn_mlp':
        targets += ['gate_proj', 'up_proj', 'down_proj']
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=targets,
        lora_dropout=0.05, bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    lora_backbone = get_peft_model(backbone, lora_cfg)
    lora_backbone.enable_input_require_grads()
    return lora_backbone, hidden_size


def batch_predict(lora_bb, head, tokenizer, texts, batch_size=4):
    device = next(lora_bb.parameters()).device
    ds = TextDataset(texts, [0.0] * len(texts), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size)
    preds = []
    lora_bb.eval(); head.eval()
    with torch.no_grad():
        for enc, _ in loader:
            enc    = {k: v.to(device) for k, v in enc.items()}
            out    = lora_bb(**enc)
            pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
            preds.extend(head(pooled).cpu().tolist())
    return preds


def staged_run(args, train_df, test_df, vd_label):
    vd_label = vd_label + args.out_suffix
    """Staged modes: 'cv' = one config on frozen folds; 'final' = full-580 fixed-epoch
    train + test eval with per-item saving and weight dumping."""
    from peft import get_peft_model_state_dict
    model_slug = args.model_name.replace('/', '_')
    split_sha  = run_utils.verify_split(train_df, test_df)
    config_id  = f"r{args.lora_r}_lr{args.lr:g}_{args.loss}"
    route      = {'text': 'Q', 'text_vd': 'Q+D'}.get(args.input_mode, vd_label)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if args.smoke:
        args.max_epochs = 1; args.epochs = 1

    if args.run_mode == 'cv':
        folds, fold_sha = run_utils.load_folds(train_df)
        assert fold_sha == split_sha, 'folds file split hash mismatch'
        sel = [int(x) for x in args.folds.split(',')]
        out_dir = os.path.join(run_utils.CV_DIR, 'llm',
                               f'{model_slug}_{vd_label}', config_id)
        os.makedirs(out_dir, exist_ok=True)
        import json as _j
        fold_results = []
        res_path = os.path.join(out_dir, 'cv_results.json')
        if os.path.exists(res_path):
            try:
                fold_results = _j.load(open(res_path)).get('fold_results', [])
                print(f"resuming: {len(fold_results)} folds already in {res_path}")
            except Exception:
                fold_results = []
        done_folds = {f['fold'] for f in fold_results}
        sel = [k for k in sel if k not in done_folds]
        for k in sel:
            tr_idx, val_idx = folds[k]
            if args.smoke:
                tr_idx = tr_idx[:40]; val_idx = val_idx[:15]
            print(f"\n=== CV fold {k}: {len(tr_idx)} train / {len(val_idx)} val "
                  f"[{config_id}] ===")
            lora_bb, hidden = build_lora_model(args.model_name, args.lora_r, args.lora_targets)
            head = RegressionHead(hidden).to(next(lora_bb.parameters()).device)
            collect = {}
            m, best_rmse = train_eval(
                args.model_name, tokenizer, lora_bb, head,
                train_df.iloc[tr_idx][args.input_mode].tolist(),
                train_df.iloc[tr_idx]['difficulty'].tolist(),
                train_df.iloc[val_idx][args.input_mode].tolist(),
                train_df.iloc[val_idx]['difficulty'].tolist(),
                lr=args.lr, batch_size=args.batch_size,
                grad_accum=args.grad_accum, max_epochs=args.max_epochs,
                patience=args.patience, loss_type=args.loss, collect=collect)
            fold_results.append({
                'fold': k, 'val_metrics': m, 'best_val_rmse': best_rmse,
                'best_epoch': collect.get('best_epoch'),
                'epoch_history': collect.get('epoch_history'),
                'val_qids': [int(q) for q in train_df.iloc[val_idx]['QuestionId']],
                'val_preds': collect.get('val_preds'),
            })
            save_results({
                'mode': 'cv', 'model': args.model_name, 'route': route,
                'config_id': config_id, 'seed': args.seed,
                'provenance': run_utils.run_provenance(args, split_sha),
                'fold_results': fold_results,
            }, os.path.join(out_dir, 'cv_results.json'))
            del lora_bb, head; torch.cuda.empty_cache()
        rmses = [f['val_metrics']['rmse'] for f in fold_results]
        print(f"\nCV done [{config_id}] folds={sel}: "
              f"mean RMSE={np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        return

    # ---- final: train on all 580, fixed epochs, no val ----
    if args.smoke:
        train_df = train_df.head(60).copy(); test_df = test_df.head(20).copy()
    out_dir = os.path.join(run_utils.MAIN_DIR, 'llm',
                           f'{model_slug}_{vd_label}', f'seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== FINAL: {args.model_name} route={route} {config_id} "
          f"epochs={args.epochs} seed={args.seed} ===")
    lora_bb, hidden = build_lora_model(args.model_name, args.lora_r, args.lora_targets)
    head = RegressionHead(hidden).to(next(lora_bb.parameters()).device)
    collect = {}
    train_eval(
        args.model_name, tokenizer, lora_bb, head,
        train_df[args.input_mode].tolist(),
        train_df['difficulty'].tolist(),
        None, None,
        lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_epochs=args.epochs, patience=0, loss_type=args.loss,
        collect=collect)

    preds = batch_predict(lora_bb, head, tokenizer,
                       test_df[args.input_mode].tolist(), args.batch_size)
    test_m = compute_metrics(test_df['difficulty'].tolist(), preds)
    print(f"TEST: RMSE={test_m['rmse']:.4f}  MAE={test_m['mae']:.4f}  "
          f"Spearman={test_m['spearman']:.4f}")

    frame = run_utils.per_item_frame(
        test_df, preds, model_id=args.model_name, route=route, seed=args.seed,
        hp_config_id=config_id, final_epoch=args.epochs,
        input_col=args.input_mode, split_sha=split_sha)
    run_utils.save_per_item(frame, out_dir)
    run_utils.save_weights(
        out_dir,
        {k: v.cpu() for k, v in get_peft_model_state_dict(lora_bb).items()},
        {k: v.cpu() for k, v in head.state_dict().items()})
    save_results({
        'mode': 'final', 'model': args.model_name, 'route': route,
        'config_id': config_id, 'seed': args.seed, 'epochs': args.epochs,
        'provenance': run_utils.run_provenance(args, split_sha),
        'epoch_history': collect.get('epoch_history'),
        'test_metrics': test_m,
    }, os.path.join(out_dir, 'results.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',  required=True)
    parser.add_argument('--input_mode',  default='text_vd', choices=['text', 'text_vd'])
    parser.add_argument('--lora_rs',     default='8,16')
    parser.add_argument('--lrs',         default='2e-4,5e-5')
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--grad_accum',  type=int, default=4)
    parser.add_argument('--max_epochs',  type=int, default=10)
    parser.add_argument('--patience',    type=int, default=3)
    parser.add_argument('--val_frac',    type=float, default=0.2)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--q_json',      default='', help='replace the question text column with this {QuestionId: text} mapping (e.g. raw OCR)')
    parser.add_argument('--q_tag',       default='', help='short name for the Q source, used in the output dir')
    parser.add_argument('--vd_json',     default='', help='path to a VLM-generated descriptions json (data/descriptions/*.json); builds Q+D from that source')
    parser.add_argument('--vd_tag',      default='', help='short name of the D source, used in the output directory name')
    # ---- staged evaluation ----
    parser.add_argument('--run_mode',     default='', choices=['', 'cv', 'final'])
    parser.add_argument('--folds',       default='0,1,2,3,4', help='cv mode: which frozen folds to run')
    parser.add_argument('--lora_r',      type=int, default=16, help='staged run: single LoRA rank')
    parser.add_argument('--lr',          type=float, default=1e-4, help='staged run: single learning rate')
    parser.add_argument('--loss',        default='mse', choices=['mse', 'huber'])
    parser.add_argument('--epochs',      type=int, default=5, help='final mode: fixed epoch count')
    parser.add_argument('--smoke',       action='store_true')
    parser.add_argument('--lora_targets', default='attn', choices=['attn', 'attn_mlp'])
    parser.add_argument('--out_suffix',  default='', help='suffix for the output directory name (ablation arms)')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    lora_rs    = [int(x)   for x in args.lora_rs.split(',')]
    lrs        = [float(x) for x in args.lrs.split(',')]
    model_slug = args.model_name.replace('/', '_')
    train_df, test_df = load_data()

    vd_label = args.input_mode
    if args.q_json:
        import json as _json2
        _qmap = {int(k): (v or '') for k, v in _json2.load(open(args.q_json)).items()}
        for _df in (train_df, test_df):
            _df['text'] = [_qmap.get(int(q), '') for q in _df['QuestionId']]
        vd_label = args.input_mode + '_q' + (args.q_tag or 'src')
        print(f"[q_json] question text replaced from {args.q_json} "
              f"({sum(1 for v in _qmap.values() if v)} non-empty)")
    if args.vd_json:
        import json as _json
        _payload = _json.load(open(args.vd_json))
        _desc = _payload.get('descriptions', _payload)
        _desc = {int(k): (v or '') for k, v in _desc.items()}
        def _mk(df):
            df = df.copy()
            df['text_vd_src'] = df.apply(
                lambda r: (str(r['text']) + ' [Figure: ' + _desc[int(r['QuestionId'])] + ']')
                          if _desc.get(int(r['QuestionId']), '') else str(r['text']),
                axis=1)
            return df
        train_df, test_df = _mk(train_df), _mk(test_df)
        args.input_mode = 'text_vd_src'
        vd_label = 'vd_' + (args.vd_tag or 'src')
        print(f"Q+D from {args.vd_json} ({sum(1 for v in _desc.values() if v)} non-empty descriptions)")

    if args.run_mode:
        staged_run(args, train_df, test_df, vd_label)
        return

    out_dir    = os.path.join(RESULTS_DIR, 'hp_search', 'llm',
                              f'{model_slug}_{vd_label}')

    # Stratified holdout used for the hyperparameter grid
    disc    = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    bins    = disc.fit_transform(train_df[['difficulty']]).ravel().astype(int)
    sss     = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, val_idx = next(sss.split(train_df, bins))

    hp_train_texts  = train_df.iloc[tr_idx][args.input_mode].tolist()
    hp_train_labels = train_df.iloc[tr_idx]['difficulty'].tolist()
    hp_val_texts    = train_df.iloc[val_idx][args.input_mode].tolist()
    hp_val_labels   = train_df.iloc[val_idx]['difficulty'].tolist()
    print(f"HP search split: {len(tr_idx)} train / {len(val_idx)} val (stratified by difficulty)")

    print(f"\n{'='*60}")
    print(f"L2-LoRA: {args.model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    search       = []
    best_val_rmse = float('inf')
    best_config   = None

    for lora_r in lora_rs:
        for lr in lrs:
            print(f"\n  [lora_r={lora_r}, lr={lr}]")
            lora_bb, hidden_size = build_lora_model(args.model_name, lora_r)
            device = next(lora_bb.parameters()).device
            head   = RegressionHead(hidden_size).to(device)

            m, val_rmse = train_eval(
                args.model_name, tokenizer, lora_bb, head,
                hp_train_texts, hp_train_labels,
                hp_val_texts, hp_val_labels,
                lr=lr, batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                max_epochs=args.max_epochs, patience=args.patience,
            )
            print(f"  val: RMSE={m['rmse']:.4f}  Sp={m['spearman']:.4f}")
            search.append({'lora_r': lora_r, 'lr': lr, 'val_metrics': m})

            if m['rmse'] < best_val_rmse:
                best_val_rmse = m['rmse']
                best_config   = {'lora_r': lora_r, 'lr': lr}

            del lora_bb, head; torch.cuda.empty_cache()

    print(f"\nBest config: {best_config}  (val RMSE={best_val_rmse:.4f})")

    # Retrain on ~90% of train; the remaining 10% is the early-stopping reference
    print("Retraining on 90% train + 10% ES split...")
    bins_all = disc.fit_transform(train_df[['difficulty']]).ravel().astype(int)
    sss_es   = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed + 1)
    ft_idx, es_idx = next(sss_es.split(train_df, bins_all))

    lora_bb, hidden_size = build_lora_model(args.model_name, best_config['lora_r'])
    device = next(lora_bb.parameters()).device
    head   = RegressionHead(hidden_size).to(device)

    train_eval(
        args.model_name, tokenizer, lora_bb, head,
        train_df.iloc[ft_idx][args.input_mode].tolist(),
        train_df.iloc[ft_idx]['difficulty'].tolist(),
        train_df.iloc[es_idx][args.input_mode].tolist(),
        train_df.iloc[es_idx]['difficulty'].tolist(),
        lr=best_config['lr'], batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_epochs=args.max_epochs, patience=args.patience,
    )

    # Test eval
    lora_bb.eval(); head.eval()
    test_ds = TextDataset(test_df[args.input_mode].tolist(), test_df['difficulty'].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    preds, trues = [], []
    with torch.no_grad():
        for enc, labels in test_loader:
            enc    = {k: v.to(device) for k, v in enc.items()}
            out    = lora_bb(**enc)
            pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
            preds.extend(head(pooled).cpu().tolist())
            trues.extend(labels.tolist())
    test_m = compute_metrics(trues, preds)
    print(f"TEST: RMSE={test_m['rmse']:.4f}  MAE={test_m['mae']:.4f}  "
          f"Pearson={test_m['pearson']:.4f}  Spearman={test_m['spearman']:.4f}")

    save_results({
        'model': args.model_name, 'input_mode': args.input_mode,
        'val_method': 'stratified_holdout_20pct',
        'best_config': best_config,
        'best_val_rmse': best_val_rmse,
        'final_train_frac': 0.9,
        'search_results': search, 'test_metrics': test_m,
    }, os.path.join(out_dir, 'results.json'))


if __name__ == '__main__':
    main()
