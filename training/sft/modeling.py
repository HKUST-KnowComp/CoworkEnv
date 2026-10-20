"""Lazy model loading for the selected text/tool SFT path."""


def load_model_and_processor(model_name, *, model_kind, dtype, revision=None, local_files_only=False):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )
    options = {'revision': revision, 'local_files_only': local_files_only, 'trust_remote_code': False}
    if model_kind == 'qwen3-vl':
        processor = AutoProcessor.from_pretrained(model_name, **options)
        cls = Qwen3VLForConditionalGeneration
        tokenizer = processor.tokenizer
    elif model_kind == 'causal-lm':
        processor = tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
        cls = AutoModelForCausalLM
    else:
        raise ValueError('Unsupported model kind')
    if not tokenizer.chat_template:
        raise ValueError('Model tokenizer must supply its own chat template')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer has neither pad nor EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = cls.from_pretrained(model_name, dtype=getattr(torch, dtype), **options)
    return model, processor, tokenizer
