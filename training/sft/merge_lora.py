"""Merge a LoRA adapter into an explicitly supplied base model on CPU.

Adapted from train/merge_lora.py. No implicit private base-model path is read
from adapter metadata. The processor/tokenizer is saved with the merged model.
"""
import argparse
from pathlib import Path
from training.sft.modeling import load_model_and_processor
from training.sft.validate_data import refuse_nonempty_output


def merge_adapter(base, adapter, output, *, model_kind='qwen3-vl', dtype='float32', revision=None, local_files_only=False):
    refuse_nonempty_output(output)
    if not (Path(adapter) / 'adapter_config.json').is_file():
        raise FileNotFoundError('Expected a local LoRA adapter directory')
    from peft import PeftModel
    model, processor, _ = load_model_and_processor(
        base, model_kind=model_kind, dtype=dtype, revision=revision,
        local_files_only=local_files_only)
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, help='Exact base used to train this adapter')
    p.add_argument('--adapter', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-kind', choices=['qwen3-vl', 'causal-lm'], default='qwen3-vl')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--revision')
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    merge_adapter(a.base, a.adapter, a.output, model_kind=a.model_kind, dtype=a.dtype,
                  revision=a.revision, local_files_only=a.local_files_only)
    print('Merged model saved.')


if __name__ == '__main__':
    main()
