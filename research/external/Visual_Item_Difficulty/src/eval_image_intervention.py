"""Visual-information ablation (eval-time intervention).

Loads a trained final-mode run (LoRA adapter + regression head saved on disk) and
re-evaluates the 145 test items with the image replaced: `blank` substitutes a
white canvas, `shuffle` pairs each item with another item's image. Per-item
predictions are written alongside the original run. No retraining involved.

Usage:
  python eval_image_intervention.py \
      --run_dir results/main/vlm/<model>_<route>/seed17 --intervention blank
"""
import os
# Set HF_HOME in your shell to control where model weights are cached.
import argparse, json, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import vlm_lora_regression as b6
from vdiff_utils import load_data, compute_metrics, save_results, resolve_image
import run_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--intervention', default='blank', choices=['blank', 'shuffle', 'none'])
    args = ap.parse_args()

    rj = json.load(open(os.path.join(args.run_dir, 'results.json')))
    ra = rj['provenance']['args']
    model_name = rj['model']
    family = b6.detect_model_family(model_name)
    b6._TEXT_FIELD = ra.get('text_field', 'text_vd')
    b6._POOL = ra.get('pool', 'all')
    include_text = ra['input_mode'] == 'image_text'

    train_df, test_df = load_data()
    split_sha = run_utils.verify_split(train_df, test_df)

    test_rows = []
    for r in test_df.to_dict('records'):
        rr = dict(r)
        if not include_text:
            rr[b6._TEXT_FIELD] = ''
        test_rows.append(rr)
    orig_images = [Image.open(resolve_image(r['image_path'])).convert('RGB')
                   for r in test_df.to_dict('records')]
    if args.intervention == 'blank':
        test_images = [Image.new('RGB', (448, 448), 'white')] * len(test_rows)
    elif args.intervention == 'shuffle':
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(orig_images))
        while (perm == np.arange(len(perm))).any():
            rng.shuffle(perm)
        test_images = [orig_images[i] for i in perm]
    else:
        test_images = orig_images

    lora_model, processor, hidden = b6.load_base_model(
        model_name, family, ra['lora_r'],
        ra.get('lora_scope', 'llm'), ra.get('lora_targets', 'attn'))
    from peft import set_peft_model_state_dict
    adapter = torch.load(os.path.join(args.run_dir, 'lora_adapter.pt'), map_location='cpu')
    set_peft_model_state_dict(lora_model, adapter)
    head = b6.RegressionHead(hidden).to('cuda')
    head.load_state_dict(torch.load(os.path.join(args.run_dir, 'regression_head.pt'),
                                    map_location='cuda'))
    lora_model.eval(); head.eval()

    preds = []
    with torch.no_grad():
        for row, img in zip(test_rows, test_images):
            text = row.get(b6._TEXT_FIELD, '') or b6.NEUTRAL_PROMPT
            pooled = b6.encode_single(lora_model, processor, family, img, text, 'cuda')
            preds.append(head(pooled).cpu().item())

    m = compute_metrics(test_df['difficulty'].tolist(), preds)
    fig = (test_df['visual_description'].fillna('') != '').to_numpy()
    gold = test_df['difficulty'].to_numpy(dtype=float)
    err = (np.array(preds) - gold) ** 2
    m['rmse_figure'] = float(np.sqrt(err[fig].mean()))
    m['rmse_nonfigure'] = float(np.sqrt(err[~fig].mean()))
    print(f"[{args.intervention}] RMSE={m['rmse']:.4f} sp={m['spearman']:.4f} "
          f"fig={m['rmse_figure']:.4f} nonfig={m['rmse_nonfigure']:.4f}")

    frame = run_utils.per_item_frame(
        test_df, preds, model_id=model_name, route=rj['route'] + f"@{args.intervention}",
        seed=rj['seed'], hp_config_id=rj['config_id'], final_epoch=rj['epochs'],
        input_col=b6._TEXT_FIELD if include_text else '', split_sha=split_sha)
    run_utils.save_per_item(frame, args.run_dir, name=f'intervention_{args.intervention}')
    save_results({'mode': 'intervention_eval', 'intervention': args.intervention,
                  'base_run': args.run_dir, 'metrics': m},
                 os.path.join(args.run_dir, f'intervention_{args.intervention}.json'))


if __name__ == '__main__':
    main()
