"""Image-native VLM: LoRA + regression head on mean-pooled last hidden states.

This is the predictor for the image-native (I+Q) interface, and the same staged
modes as llm_lora_regression.py apply (--run_mode cv / final, seeds 17/42/2026).

Key switches:
  --input_mode  image_only | image_text   (I vs I+Q)
  --text_field  text | text_vd            (Q vs Q+D alongside the image)
  --lora_targets attn | attn_mlp          (adaptation breadth, Table 4)
  --lora_scope  llm | llm_vision          (also adapt the vision encoder)
  --pool        all | text                (pool over all tokens vs text tokens only)

Supports the Qwen-VL, InternVL and PaliGemma families.

Usage:
    CUDA_VISIBLE_DEVICES=0 python vlm_lora_regression.py \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --input_mode image_text \
        --text_field text --lora_targets attn_mlp \
        --run_mode final --lora_r 16 --lr 2e-4 --epochs 5 --seed 42
"""

import os
# Set HF_HOME in your shell to control where model weights are cached.
# Gated checkpoints need HF_TOKEN exported in your shell.

import argparse, sys
import numpy as np
import torch
# transformers>=5 expects model.all_tied_weights_keys; InternVL's trust_remote_code
# modeling code (older) only defines _tied_weights_keys. Provide an empty-dict fallback
# so caching_allocator_warmup/get_total_byte_count doesn't crash on InternVLChatModel.
from transformers.modeling_utils import PreTrainedModel as _PTM
if not hasattr(_PTM, 'all_tied_weights_keys'):
    _PTM.all_tied_weights_keys = {}
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, os.path.dirname(__file__))
from vdiff_utils import load_data, compute_metrics, save_results, RESULTS_DIR, resolve_image
import run_utils
_TEXT_FIELD = 'text_vd'  # which text column image_text uses; set from --text_field
_POOL = 'all'            # 'all' = mean over full sequence; 'text' = exclude image tokens
_IMG_IDS = set()         # token ids treated as image placeholders for _POOL='text'

NEUTRAL_PROMPT = "Represent this math assessment item for predicting student difficulty. Do not solve it."


def detect_model_family(model_name):
    name = model_name.lower()
    if 'qwen' in name and ('vl' in name or 'vision' in name):
        return 'qwen-vl'
    elif 'internvl' in name:
        return 'internvl'
    elif 'paligemma' in name:
        return 'paligemma'
    else:
        return 'generic'


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


VISION_KEYS = ('visual', 'vision_tower', 'vision_model')
VISION_LEAFS = ('qkv', 'proj', 'q_proj', 'k_proj', 'v_proj', 'out_proj')


def vision_linear_names(model):
    """Full paths of vision-encoder attention Linears (for LoRA scope=llm_vision)."""
    import torch.nn as _nn
    names = []
    for n, mod in model.named_modules():
        if not isinstance(mod, _nn.Linear):
            continue
        if any(k in n for k in VISION_KEYS) and n.rsplit('.', 1)[-1] in VISION_LEAFS:
            names.append(n)
    return names


def collect_image_token_ids(family, processor, base):
    ids = set()
    tok = getattr(processor, 'tokenizer', processor)
    if family in ('qwen-vl', 'generic'):
        for t in ('<|image_pad|>', '<|vision_start|>', '<|vision_end|>'):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i >= 0:
                ids.add(i)
    elif family == 'internvl':
        for t in ('<IMG_CONTEXT>', '<img>', '</img>'):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i >= 0:
                ids.add(i)
    elif family == 'paligemma':
        i = getattr(base.config, 'image_token_index', None)
        if i is not None:
            ids.add(int(i))
    return ids


def load_base_model(model_name, family, lora_r, lora_scope='llm', lora_targets='attn'):
    from peft import LoraConfig, get_peft_model, TaskType

    if family == 'qwen-vl':
        from transformers import AutoProcessor
        if 'qwen3' in model_name.lower():
            from transformers import AutoModelForImageTextToText
            base = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map='auto')
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map='auto')
            except (ImportError, AttributeError):
                from transformers import AutoModelForVision2Seq
                base = AutoModelForVision2Seq.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    elif family == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        base = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={'': 0},
            trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        # peft's wrapper forward injects inputs_embeds=None, which InternVL's
        # custom forward() rejects; patch the class to swallow it.
        _ivc = type(base)
        if not getattr(_ivc, '_iv_embeds_patched', False):
            _of = _ivc.forward
            def _pf(self, *a, inputs_embeds=None, **kw):
                return _of(self, *a, **kw)
            _ivc.forward = _pf
            _ivc._iv_embeds_patched = True

    elif family == 'paligemma':
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        base = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)
        if getattr(processor, 'tokenizer', None) is not None and processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if lora_targets == 'attn_mlp':
        targets += ['gate_proj', 'up_proj', 'down_proj']
    if lora_scope == 'llm_vision':
        vis = vision_linear_names(base)
        targets = targets + vis
        print(f"[lora_scope=llm_vision] +{len(vis)} vision-encoder linears")
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=targets,
        lora_dropout=0.05, bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    lora_model = get_peft_model(base, lora_cfg)
    lora_model.enable_input_require_grads()
    global _IMG_IDS
    _IMG_IDS = collect_image_token_ids(family, processor, base)

    # The pooled feature is out.hidden_states[-1] (the LLM's last hidden state),
    # so we need the *text/LLM* hidden size. For PaliGemma the top-level
    # config.hidden_size (2048) is NOT the LLM size (Gemma2 = 2304); prefer
    # text_config / llm_config and only fall back to the top-level value.
    cfg = base.config
    if family == 'internvl':
        cands = [getattr(getattr(cfg, 'llm_config', None), 'hidden_size', None),
                 getattr(getattr(cfg, 'text_config', None), 'hidden_size', None)]
    else:  # qwen-vl, paligemma, generic
        cands = [getattr(getattr(cfg, 'text_config', None), 'hidden_size', None),
                 getattr(getattr(cfg, 'llm_config', None), 'hidden_size', None)]
    cands.append(getattr(cfg, 'hidden_size', None))
    hidden_size = next((c for c in cands if c), None)
    if hidden_size is None:
        raise ValueError(f'could not detect hidden_size for family={family}')
    return lora_model, processor, hidden_size


def encode_single(model, processor, family, image, text, device):
    """Forward pass a single item and return mean-pooled last hidden state."""
    if family == 'qwen-vl':
        content = [{"type": "image", "image": image}, {"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        try:
            from qwen_vl_utils import process_vision_info
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            inp = processor(text=[text_input], images=image_inputs,
                            return_tensors='pt', padding=True)
        except ImportError:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inp = processor(text=[text_input], images=[image],
                            return_tensors='pt', padding=True)

    elif family == 'paligemma':
        inp = processor(text=text, images=image, return_tensors='pt', padding=True)

    elif family == 'internvl':
        import torchvision.transforms as T
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize((448, 448)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16)
        # InternVLChatModel.forward needs IMG_CONTEXT placeholders + image_flags.
        base_iv = model.get_base_model()
        IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
        base_iv.img_context_token_id = processor.convert_tokens_to_ids(IMG_CTX)
        image_tokens = IMG_START + IMG_CTX * base_iv.num_image_token + IMG_END
        enc = processor(f"{image_tokens}\n{text}", return_tensors='pt',
                        truncation=True, max_length=2048)
        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)
        inp = {'pixel_values': pixel_values, 'image_flags': image_flags, **enc}

    else:
        content = [{"type": "image", "image": image}, {"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        if hasattr(processor, 'apply_chat_template'):
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inp = processor(text=[text_input], images=[image], return_tensors='pt')
        else:
            inp = processor(images=[image], text=[text], return_tensors='pt')

    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    hidden = out.hidden_states[-1]  # [1, seq_len, hidden]
    if _POOL == 'text' and 'input_ids' in inp and _IMG_IDS:
        ids = inp['input_ids'][0]
        keep = torch.ones_like(ids, dtype=torch.bool)
        for tid in _IMG_IDS:
            keep &= ids != tid
        if keep.shape[0] == hidden.shape[1] and keep.any():
            pooled = hidden[0][keep].float().mean(dim=0, keepdim=True)
            return pooled
    pooled = hidden.float().mean(dim=1)  # [1, hidden]
    return pooled


def compute_loss(preds, labels, loss_type):
    if loss_type == 'huber':
        return F.huber_loss(preds, labels, delta=1.0)
    else:
        return F.mse_loss(preds, labels)


def train_eval_regression(lora_model, head, processor, family,
                          tr_rows, tr_imgs, val_rows, val_imgs,
                          device, lr, loss_type, batch_size=1, grad_accum=4,
                          max_epochs=10, patience=3, collect=None):
    """Early-stopping train/val; if val_rows is None, train exactly max_epochs
    on all data with no validation (final-train mode)."""
    no_val = val_rows is None
    trainable_bb = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_bb + list(head.parameters()), lr=lr, weight_decay=0.01)
    total_steps = (len(tr_rows) // (batch_size * grad_accum) + 1) * max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps)

    best_rmse = float('inf')
    best_bb_state = None
    best_head_state = None
    best_epoch = 0
    no_improve = 0
    epoch_history = []

    for epoch in range(max_epochs):
        lora_model.train()
        head.train()
        optimizer.zero_grad()
        epoch_loss, n_items = 0.0, 0

        # Shuffle
        indices = np.random.permutation(len(tr_rows))
        for step_i, idx in enumerate(indices):
            row = tr_rows[idx]
            img = tr_imgs[idx]
            label = torch.tensor([row['difficulty']], dtype=torch.float32).to(device)

            text = row.get(_TEXT_FIELD, '') or NEUTRAL_PROMPT
            pooled = encode_single(lora_model, processor, family, img, text, device)
            pred = head(pooled)
            loss = compute_loss(pred, label, loss_type) / grad_accum

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            epoch_loss += loss.item() * grad_accum; n_items += 1
            if (step_i + 1) % grad_accum == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(
                    trainable_bb + list(head.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if no_val:
            tr_loss = epoch_loss / max(1, n_items)
            epoch_history.append({'epoch': epoch + 1, 'train_loss': tr_loss})
            print(f"    epoch {epoch+1}: train loss={tr_loss:.4f}")
            continue

        # Validation
        lora_model.eval()
        head.eval()
        preds, trues = [], []
        with torch.no_grad():
            for row, img in zip(val_rows, val_imgs):
                text = row.get(_TEXT_FIELD, '') or NEUTRAL_PROMPT
                pooled = encode_single(lora_model, processor, family, img, text, device)
                pred = head(pooled).cpu().item()
                preds.append(pred)
                trues.append(row['difficulty'])

        val_rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2)))
        print(f"    epoch {epoch+1}: val RMSE={val_rmse:.4f}")
        epoch_history.append({'epoch': epoch + 1,
                              'train_loss': epoch_loss / max(1, n_items),
                              'val_rmse': val_rmse})

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            from peft import get_peft_model_state_dict
            best_bb_state = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(lora_model).items()}
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

    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(lora_model, best_bb_state)
    head.load_state_dict(best_head_state)
    lora_model.eval()
    head.eval()

    preds, trues = [], []
    with torch.no_grad():
        for row, img in zip(val_rows, val_imgs):
            text = row.get(_TEXT_FIELD, '') or NEUTRAL_PROMPT
            pooled = encode_single(lora_model, processor, family, img, text, device)
            preds.append(head(pooled).cpu().item())
            trues.append(row['difficulty'])
    if collect is not None:
        collect['epoch_history'] = epoch_history
        collect['best_epoch']    = best_epoch
        collect['val_preds']     = preds
        collect['val_trues']     = trues
    return compute_metrics(trues, preds), best_rmse


def route_tag(args):
    if args.input_mode == 'image_only':
        return 'I'
    return 'I+Q' if args.text_field == 'text' else 'I+Q+D'


def staged_run_vlm(args, family, train_df, test_df, train_rows, test_rows,
            train_images, test_images, device):
    """Staged modes: 'cv' = one config on frozen folds;
    'final' = full-580 fixed-epoch train + per-item test predictions."""
    from peft import get_peft_model_state_dict
    model_slug = args.model_name.replace('/', '_')
    split_sha  = run_utils.verify_split(train_df, test_df)
    config_id  = f"r{args.lora_r}_lr{args.lr:g}_{args.loss}"
    route      = route_tag(args)
    route_tag  = route.replace('+', '') + args.out_suffix

    if args.smoke:
        args.max_epochs = 1; args.epochs = 1

    if args.run_mode == 'cv':
        folds, fold_sha = run_utils.load_folds(train_df)
        assert fold_sha == split_sha, 'folds file split hash mismatch'
        sel = [int(x) for x in args.folds.split(',')]
        out_dir = os.path.join(run_utils.CV_DIR, 'vlm',
                               f'{model_slug}_{route_tag}', config_id)
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
                tr_idx = tr_idx[:30]; val_idx = val_idx[:10]
            print(f"\n=== CV fold {k}: {len(tr_idx)} train / {len(val_idx)} val "
                  f"[{config_id}] route={route} ===")
            lora_model, processor, hidden = load_base_model(
                args.model_name, family, args.lora_r,
                args.lora_scope, args.lora_targets)
            head = RegressionHead(hidden).to(device)
            collect = {}
            m, best_rmse = train_eval_regression(
                lora_model, head, processor, family,
                [train_rows[i] for i in tr_idx], [train_images[i] for i in tr_idx],
                [train_rows[i] for i in val_idx], [train_images[i] for i in val_idx],
                device, lr=args.lr, loss_type=args.loss,
                grad_accum=args.grad_accum, max_epochs=args.max_epochs,
                patience=args.patience, collect=collect)
            fold_results.append({
                'fold': k, 'val_metrics': m, 'best_val_rmse': best_rmse,
                'best_epoch': collect.get('best_epoch'),
                'epoch_history': collect.get('epoch_history'),
                'val_qids': [int(train_rows[i]['QuestionId']) for i in val_idx],
                'val_preds': collect.get('val_preds'),
            })
            save_results({
                'mode': 'cv', 'model': args.model_name, 'family': family,
                'route': route, 'config_id': config_id, 'seed': args.seed,
                'provenance': run_utils.run_provenance(args, split_sha),
                'fold_results': fold_results,
            }, os.path.join(out_dir, 'cv_results.json'))
            del lora_model, head; torch.cuda.empty_cache()
        rmses = [f['val_metrics']['rmse'] for f in fold_results]
        print(f"\nCV done [{config_id}] folds={sel}: "
              f"mean RMSE={np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        return

    # ---- final: train on all 580, fixed epochs, no val ----
    if args.smoke:
        train_rows = train_rows[:30]; train_images = train_images[:30]
        test_df = test_df.head(10).copy()
        test_rows = test_rows[:10]; test_images = test_images[:10]
    out_dir = os.path.join(run_utils.MAIN_DIR, 'vlm',
                           f'{model_slug}_{route_tag}', f'seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== FINAL: {args.model_name} route={route} {config_id} "
          f"epochs={args.epochs} seed={args.seed} ===")
    lora_model, processor, hidden = load_base_model(
        args.model_name, family, args.lora_r,
        args.lora_scope, args.lora_targets)
    head = RegressionHead(hidden).to(device)
    collect = {}
    train_eval_regression(
        lora_model, head, processor, family,
        train_rows, train_images, None, None,
        device, lr=args.lr, loss_type=args.loss,
        grad_accum=args.grad_accum, max_epochs=args.epochs,
        patience=0, collect=collect)

    lora_model.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for row, img in zip(test_rows, test_images):
            text = row.get(_TEXT_FIELD, '') or NEUTRAL_PROMPT
            pooled = encode_single(lora_model, processor, family, img, text, device)
            preds.append(head(pooled).cpu().item())
    test_m = compute_metrics([r['difficulty'] for r in test_rows], preds)
    print(f"TEST: RMSE={test_m['rmse']:.4f}  MAE={test_m['mae']:.4f}  "
          f"Spearman={test_m['spearman']:.4f}")

    input_col = _TEXT_FIELD if args.input_mode == 'image_text' else ''
    frame = run_utils.per_item_frame(
        test_df, preds, model_id=args.model_name, route=route, seed=args.seed,
        hp_config_id=config_id, final_epoch=args.epochs,
        input_col=input_col, split_sha=split_sha)
    run_utils.save_per_item(frame, out_dir)
    run_utils.save_weights(
        out_dir,
        {k: v.cpu() for k, v in get_peft_model_state_dict(lora_model).items()},
        {k: v.cpu() for k, v in head.state_dict().items()})
    save_results({
        'mode': 'final', 'model': args.model_name, 'family': family,
        'route': route, 'config_id': config_id, 'seed': args.seed,
        'epochs': args.epochs,
        'provenance': run_utils.run_provenance(args, split_sha),
        'epoch_history': collect.get('epoch_history'),
        'test_metrics': test_m,
    }, os.path.join(out_dir, 'results.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--input_mode', choices=['image_only', 'image_text'],
                        default='image_only')
    parser.add_argument('--lora_rs', default='8,16,32')
    parser.add_argument('--lrs', default='2e-4,5e-5')
    parser.add_argument('--losses', default='mse,huber')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny pre-flight: 60 train/20 test, 1 HP config, 1 epoch')
    parser.add_argument('--out_suffix', default='',
                        help='Suffix appended to mode_tag output dir, e.g. _ep5 / _imgQ')
    parser.add_argument('--text_field', choices=['text_vd', 'text'], default='text_vd',
                        help='text_vd = Q+D (default); text = Q only (image+Q ablation)')
    # ---- staged evaluation ----
    parser.add_argument('--run_mode', default='', choices=['', 'cv', 'final'])
    parser.add_argument('--folds', default='0,1,2,3,4', help='cv mode: which frozen folds to run')
    parser.add_argument('--lora_r', type=int, default=16, help='staged run: single LoRA rank')
    parser.add_argument('--lr', type=float, default=2e-4, help='staged run: single learning rate')
    parser.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    parser.add_argument('--epochs', type=int, default=5, help='final mode: fixed epoch count')
    parser.add_argument('--lora_scope', default='llm', choices=['llm', 'llm_vision'],
                        help='llm = adapt LLM only (vision frozen); llm_vision = also LoRA the vision-encoder attention')
    parser.add_argument('--lora_targets', default='attn', choices=['attn', 'attn_mlp'])
    parser.add_argument('--pool', default='all', choices=['all', 'text'],
                        help='mean-pool over all tokens vs text tokens only (image tokens excluded)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    global _TEXT_FIELD, _POOL
    _TEXT_FIELD = args.text_field
    _POOL = args.pool

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lora_rs = [int(x) for x in args.lora_rs.split(',')]
    lrs = [float(x) for x in args.lrs.split(',')]
    losses = args.losses.split(',')
    if args.smoke:
        lora_rs, lrs, losses, args.max_epochs, args.patience = [8], [2e-4], ['mse'], 1, 1
        print("[SMOKE] 1 HP config (r=8, lr=2e-4, mse), 1 epoch, tiny subset")
    model_slug = args.model_name.replace('/', '_')
    mode_tag = 'R1' if args.input_mode == 'image_only' else 'R2'
    out_dir = os.path.join(RESULTS_DIR, 'hp_search', 'vlm', mode_tag.lower() + args.out_suffix, model_slug)
    include_text = args.input_mode == 'image_text'

    family = detect_model_family(args.model_name)
    print(f"\nBlock 6 / {mode_tag}: {args.model_name}  family={family}")

    train_df, test_df = load_data()

    print("Loading images...")
    import pandas as pd
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_images = []
    for _, row in all_df.iterrows():
        img_path = resolve_image(row['image_path'])
        all_images.append(Image.open(img_path).convert('RGB'))
    train_images = all_images[:len(train_df)]
    test_images = all_images[len(train_df):]

    # For image_only mode, strip text from rows to avoid accidental use
    def prep_row(row_dict, include_txt):
        r = dict(row_dict)
        if not include_txt:
            r[_TEXT_FIELD] = ''
        return r

    train_rows = [prep_row(r, include_text) for r in train_df.to_dict('records')]
    test_rows = [prep_row(r, include_text) for r in test_df.to_dict('records')]

    if args.run_mode:
        staged_run_vlm(args, family, train_df, test_df, train_rows, test_rows,
                train_images, test_images, device)
        return

    if args.smoke:
        train_df = train_df.head(60); test_df = test_df.head(20)
        train_images = train_images[:60]; test_images = test_images[:20]
        train_rows = train_rows[:60]; test_rows = test_rows[:20]

    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    bins = disc.fit_transform(train_df[['difficulty']]).ravel().astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, val_idx = next(sss.split(train_df, bins))
    hp_tr_rows = [train_rows[i] for i in tr_idx]
    hp_tr_imgs = [train_images[i] for i in tr_idx]
    hp_val_rows = [train_rows[i] for i in val_idx]
    hp_val_imgs = [train_images[i] for i in val_idx]
    print(f"HP search: {len(tr_idx)} train / {len(val_idx)} val")

    print(f"\n{'='*60}")

    search_results = []
    best_val_rmse = float('inf')
    best_config = None

    import json as _json
    for lora_r in lora_rs:
        for lr in lrs:
            for loss_type in losses:
                print(f"\n  [lora_r={lora_r}, lr={lr}, loss={loss_type}]")
                try:
                    lora_model, processor, hidden_size = load_base_model(
                        args.model_name, family, lora_r,
                        args.lora_scope, args.lora_targets)
                    head = RegressionHead(hidden_size).to(device)

                    m, val_rmse = train_eval_regression(
                        lora_model, head, processor, family,
                        hp_tr_rows, hp_tr_imgs, hp_val_rows, hp_val_imgs,
                        device, lr=lr, loss_type=loss_type,
                        grad_accum=args.grad_accum, max_epochs=args.max_epochs,
                        patience=args.patience)
                    print(f"  val: RMSE={m['rmse']:.4f}  Sp={m['spearman']:.4f}")
                    search_results.append({
                        'lora_r': lora_r, 'lr': lr, 'loss': loss_type,
                        'val_metrics': m, 'val_rmse': val_rmse, 'status': 'ok'})
                    if m['rmse'] < best_val_rmse:
                        best_val_rmse = m['rmse']
                        best_config = {'lora_r': lora_r, 'lr': lr, 'loss': loss_type}
                except Exception as _e:
                    print(f"  FAILED: {_e}")
                    search_results.append({
                        'lora_r': lora_r, 'lr': lr, 'loss': loss_type,
                        'status': 'failed', 'error': str(_e)})
                finally:
                    try:
                        del lora_model, head
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                # Save progress after each candidate
                _partial_path = os.path.join(out_dir, 'hp_search_progress.json')
                os.makedirs(out_dir, exist_ok=True)
                with open(_partial_path, 'w') as _f:
                    _json.dump({'search_results': search_results, 'best_config': best_config}, _f, indent=2, default=float)

    print(f"\nBest config: {best_config}  (val RMSE={best_val_rmse:.4f})")

    # Retrain on 90%
    sss_es = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed + 1)
    ft_idx, es_idx = next(sss_es.split(train_df, bins))
    ft_rows = [train_rows[i] for i in ft_idx]
    ft_imgs = [train_images[i] for i in ft_idx]
    es_rows = [train_rows[i] for i in es_idx]
    es_imgs = [train_images[i] for i in es_idx]

    print("\nRetraining on 90% train...")
    lora_model, processor, hidden_size = load_base_model(
        args.model_name, family, best_config['lora_r'],
        args.lora_scope, args.lora_targets)
    head = RegressionHead(hidden_size).to(device)

    train_eval_regression(
        lora_model, head, processor, family,
        ft_rows, ft_imgs, es_rows, es_imgs,
        device, lr=best_config['lr'], loss_type=best_config['loss'],
        grad_accum=args.grad_accum, max_epochs=args.max_epochs,
        patience=args.patience)

    # Test evaluation
    lora_model.eval()
    head.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for row, img in zip(test_rows, test_images):
            text = row.get(_TEXT_FIELD, '') or NEUTRAL_PROMPT
            pooled = encode_single(lora_model, processor, family, img, text, device)
            test_preds.append(head(pooled).cpu().item())
            test_trues.append(row['difficulty'])

    test_m = compute_metrics(test_trues, test_preds)
    print(f"TEST: RMSE={test_m['rmse']:.4f}  MAE={test_m['mae']:.4f}  "
          f"Pearson={test_m['pearson']:.4f}  Spearman={test_m['spearman']:.4f}")

    save_results({
        'model': args.model_name, 'family': family,
        'input_mode': args.input_mode, 'mode_tag': mode_tag,
        'val_method': 'stratified_holdout_20pct',
        'best_config': best_config, 'best_val_rmse': best_val_rmse,
        'search_results': search_results,
        'test_metrics': test_m,
    }, os.path.join(out_dir, 'results.json'))


if __name__ == '__main__':
    main()
