"""Per-family VLM loading and generation helpers.

Covers the Qwen-VL, InternVL and PaliGemma families used in the paper, hiding the
processor / chat-template / image-token differences behind a common interface:

    family = detect_model_family(model_name)
    model, processor = load_model_and_processor(model_name, family, lora_r=16)
    text = generate_prediction(model, processor, family, image, prompt, device)

Used by generate_descriptions.py; vlm_lora_regression.py carries its own copy of
the loading logic because it also needs hidden-state access for the regression head.
"""
import os, re

import torch
# transformers>=5 expects model.all_tied_weights_keys; InternVL's trust_remote_code
# modeling code (older) only defines _tied_weights_keys. Provide an empty-dict
# fallback so caching_allocator_warmup does not crash on InternVLChatModel.
from transformers.modeling_utils import PreTrainedModel as _PTM
if not hasattr(_PTM, 'all_tied_weights_keys'):
    _PTM.all_tied_weights_keys = {}

_IV_NUM_IMG_TOKEN = 256  # InternVL2.5 @448px; overwritten at model load


def detect_model_family(model_name):
    name = model_name.lower()
    if 'qwen' in name and ('vl' in name or 'vision' in name):
        return 'qwen-vl'
    elif 'internvl' in name:
        return 'internvl'
    elif 'paligemma' in name:
        return 'paligemma'
    return 'generic'

def load_model_and_processor(model_name, family, lora_r=16):
    from peft import LoraConfig, get_peft_model, TaskType

    if family == 'qwen-vl':
        from transformers import AutoProcessor
        if 'qwen3' in model_name.lower():
            # Qwen3-VL is a distinct architecture class; AutoModelForImageTextToText resolves it.
            from transformers import AutoModelForImageTextToText
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map='auto')
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map='auto')
            except (ImportError, AttributeError):
                from transformers import AutoModelForVision2Seq
                base_model = AutoModelForVision2Seq.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    elif family == 'internvl':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={'': 0},
            trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        # InternVL forward/generate need img_context_token_id set and IMG_CONTEXT
        # placeholders in input_ids (a literal "<image>" string does NOT work).
        global _IV_NUM_IMG_TOKEN
        base_model.img_context_token_id = processor.convert_tokens_to_ids('<IMG_CONTEXT>')
        _IV_NUM_IMG_TOKEN = base_model.num_image_token
        # peft's wrapper forward injects inputs_embeds=None, which InternVL's
        # custom forward() rejects; patch the class to swallow it.
        _ivc = type(base_model)
        if not getattr(_ivc, '_iv_embeds_patched', False):
            _of = _ivc.forward
            def _pf(self, *a, inputs_embeds=None, **kw):
                return _of(self, *a, **kw)
            _ivc.forward = _pf
            _ivc._iv_embeds_patched = True

    elif family == 'paligemma':
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    else:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05, bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()
    return model, processor

def get_tokenizer(processor, family):
    if family in ('qwen-vl', 'paligemma'):
        return processor.tokenizer
    return processor

def parse_difficulty_z(text):
    """Parse generated JSON to extract difficulty_z float."""
    try:
        obj = json.loads(text.strip())
        return float(obj.get('difficulty_z', float('nan')))
    except Exception:
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')

def generate_prediction(model, processor, family, image, text_prompt, device, max_new_tokens=30):
    """Generate difficulty_z prediction for a single item."""
    tok = get_tokenizer(processor, family)

    if family == 'qwen-vl':
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt},
        ]}]
        try:
            from qwen_vl_utils import process_vision_info
            text_in = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            img_inputs, _ = process_vision_info(messages)
            inp = processor(text=[text_in], images=img_inputs,
                            return_tensors='pt', padding=False)
        except ImportError:
            text_in = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = processor(text=[text_in], images=[image],
                            return_tensors='pt', padding=False)

    elif family == 'paligemma':
        inp = processor(text=text_prompt, images=image, return_tensors='pt', padding=False)

    elif family == 'internvl':
        import torchvision.transforms as T
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize((448, 448)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(device)
        IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
        image_tokens = IMG_START + IMG_CTX * _IV_NUM_IMG_TOKEN + IMG_END
        enc = tok(f"{image_tokens}\n{text_prompt}", return_tensors='pt')
        inp = {k: v.to(device) for k, v in enc.items()}
        inp['pixel_values'] = pixel_values

    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt},
        ]}]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text_in], images=[image], return_tensors='pt', padding=False)

    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
    # InternVL has a custom .generate (delegates to language_model.generate with
    # inputs_embeds); peft's generate wrapper instead hits the standard path which
    # calls prepare_inputs_for_generation that InternVLChatModel lacks. Route through
    # the base model so the custom generate runs (LoRA adapters stay active).
    gen_model = model.get_base_model() if family == 'internvl' and hasattr(model, 'get_base_model') else model
    with torch.no_grad():
        gen = gen_model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    # InternVL.generate() runs language_model.generate(inputs_embeds=...), which
    # returns ONLY the newly generated tokens (no input prefix); slicing by the
    # input length would drop the actual output. Other families return the full
    # sequence, so slice off the prompt for them.
    if family == 'internvl':
        new_tokens = gen
    else:
        new_tokens = gen[:, inp['input_ids'].shape[1]:]
    return tok.decode(new_tokens[0], skip_special_tokens=True)
