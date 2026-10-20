"""Train text/tool LoRA adapters with Transformers Trainer.

Adapted from train/run_sft_iter2.py. This preserves its full-conversation causal
LM objective (only padding is masked), not an assistant-only objective. Images
and videos are rejected. Qwen3-VL uses its matching model class, not Qwen2-VL.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
from training.jsonl import read_records
from training.sft.modeling import load_model_and_processor
from training.sft.validate_data import inspect_jsonl, refuse_nonempty_output, template_messages


def tokenize_record(record, tokenizer, max_length):
    options = {'tools': record['tools']} if 'tools' in record else {}
    ids = tokenizer.apply_chat_template(template_messages(record['messages']),
                                        tokenize=True, return_dict=False,
                                        add_generation_prompt=False, **options)
    if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(x, int) for x in ids):
        raise ValueError('Chat template must produce at least two token IDs')
    if len(ids) > max_length:
        raise ValueError('Record exceeds --max-length; filter/split your data explicitly rather than cutting tool pairs')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': ids.copy()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local model directory or Hugging Face model ID')
    p.add_argument('--model-kind', choices=['qwen3-vl', 'causal-lm'], default='qwen3-vl')
    p.add_argument('--revision', help='Model revision; use a commit for reproducibility')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--use-cpu', action='store_true')
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--lora-rank', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--lora-dropout', type=float, default=.05)
    p.add_argument('--target-modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    for name in ('max_length', 'epochs', 'batch_size', 'gradient_accumulation', 'learning_rate', 'lora_rank', 'lora_alpha'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            p.error(f'{name} must be finite and positive')
    if not 0 <= args.lora_dropout < 1:
        p.error('lora_dropout must be in [0, 1)')
    return args


def run(args):
    # This release deliberately exposes a single-process entrypoint. Do not
    # accidentally let ranks race when reusing an old cluster launcher.
    if int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('This entrypoint supports one process; distributed recipes are not included')
    refuse_nonempty_output(args.output)
    provenance = inspect_jsonl(args.data)
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    model, processor, tokenizer = load_model_and_processor(
        args.model, model_kind=args.model_kind, dtype=args.dtype,
        revision=args.revision, local_files_only=args.local_files_only)
    rows = [tokenize_record(record, tokenizer, args.max_length) for record in read_records(args.data)]
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias='none'))
    model.print_trainable_parameters()
    options = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=.1,
        logging_steps=1, save_strategy='epoch', save_total_limit=2,
        bf16=args.dtype == 'bfloat16', use_cpu=args.use_cpu,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, dataloader_pin_memory=not args.use_cpu,
        remove_unused_columns=False, report_to='none', seed=args.seed)
    trainer = Trainer(
        model=model, args=options, train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                           pad_to_multiple_of=8, label_pad_token_id=-100),
        processing_class=tokenizer)
    result = trainer.train()
    final = args.output / 'final'
    trainer.save_model(str(final))
    processor.save_pretrained(final)
    run_record = {
        'dataset': provenance,
        'objective': 'full_conversation_causal_lm_padding_masked',
        'model_kind': args.model_kind, 'model': args.model, 'revision': args.revision,
        'hyperparameters': {k: v for k, v in vars(args).items() if k not in {'data', 'output', 'model'}},
        'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'peft', 'accelerate', 'datasets']},
        'metrics': result.metrics,
    }
    (args.output / 'training_record.json').write_text(json.dumps(run_record, indent=2, allow_nan=False) + '\n')
    return final


def main(argv=None):
    print('Saved adapter:', run(parse_args(argv)))


if __name__ == '__main__':
    main()
