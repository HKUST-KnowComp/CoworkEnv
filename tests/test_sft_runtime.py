"""Optional CPU checks with local tiny models; no downloaded weights or API calls.

These are implementation tests, not benchmark examples or performance results.
Install requirements-sft.txt to run them.
"""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in
                ('torch', 'transformers', 'peft', 'accelerate', 'datasets'))


@unittest.skipUnless(AVAILABLE, 'Optional SFT dependencies are not installed')
class SftRuntimeTests(unittest.TestCase):
    def make_base(self, path, kind):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
                                  Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                                  Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
        torch.manual_seed(7)
        vocab = {s: i for i, s in enumerate([
            '[UNK]', '[PAD]', '[EOS]', 'user', 'assistant', 'hello', 'world',
            'again', '<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>'])}
        raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                           pad_token='[PAD]', eos_token='[EOS]')
        template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' + eos_token + ' ' }}{% endfor %}"
        tokenizer.chat_template = template
        text = dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=128, pad_token_id=1)
        if kind == 'causal-lm':
            model = Qwen3ForCausalLM(Qwen3Config(**text, eos_token_id=2))
            processor = tokenizer
        else:
            config = Qwen3VLConfig(
                text_config={**text, 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000., 'mrope_section': [1, 1, 2]}},
                vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                                   out_hidden_size=32, patch_size=2, spatial_merge_size=2,
                                   temporal_patch_size=1, num_position_embeddings=16,
                                   deepstack_visual_indexes=[]),
                image_token_id=8, video_token_id=9, vision_start_token_id=10, vision_end_token_id=11)
            model = Qwen3VLForConditionalGeneration(config)
            processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
                                        video_processor=Qwen3VLVideoProcessor(), chat_template=template)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        return tokenizer

    def test_padding_masks_only_padding_and_refuses_truncation(self):
        from transformers import DataCollatorForSeq2Seq
        from training.sft.train_lora import tokenize_record
        with tempfile.TemporaryDirectory() as d:
            tokenizer = self.make_base(Path(d), 'causal-lm')
            tokenizer.pad_token = tokenizer.eos_token
            records = [{'messages': [{'role': 'user', 'content': 'hello'},
                                     {'role': 'assistant', 'content': s}]} for s in ('world', 'world again')]
            rows = [tokenize_record(x, tokenizer, 64) for x in records]
            batch = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)(rows)
            for i, row in enumerate(rows):
                n = len(row['input_ids'])
                self.assertEqual(batch['labels'][i, :n].tolist(), row['input_ids'])
                self.assertTrue((batch['labels'][i, n:] == -100).all())
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                tokenize_record(records[0], tokenizer, 2)

    def test_train_save_and_merge_both_model_paths(self):
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from training.sft.train_lora import run, parse_args
        from training.sft.modeling import load_model_and_processor
        from training.sft.merge_lora import merge_adapter
        torch.set_num_threads(1)
        for kind in ('causal-lm', 'qwen3-vl'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                base, output, merged = root / 'base', root / 'run', root / 'merged'
                self.make_base(base, kind)
                data = root / 'data.jsonl'
                data.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'hello'},
                                                        {'role': 'assistant', 'content': 'world again'}]}) + '\n')
                args = parse_args(['--model', str(base), '--model-kind', kind, '--data', str(data),
                                   '--output', str(output), '--dtype', 'float32', '--use-cpu',
                                   '--local-files-only', '--epochs', '2', '--gradient-accumulation', '1',
                                   '--learning-rate', '.01', '--lora-rank', '2', '--lora-alpha', '4',
                                   '--lora-dropout', '0', '--max-length', '128'])
                final = run(args)
                record = json.loads((output / 'training_record.json').read_text())
                self.assertTrue(math.isfinite(record['metrics']['train_loss']))
                weights = load_file(final / 'adapter_model.safetensors')
                self.assertTrue(any(v.abs().sum() > 0 for k, v in weights.items() if 'lora_B' in k))
                with self.assertRaises(FileExistsError):
                    run(args)
                model, _, _ = load_model_and_processor(str(base), model_kind=kind, dtype='float32', local_files_only=True)
                adapted = PeftModel.from_pretrained(model, final).eval()
                inputs = torch.tensor([[3, 5, 2, 4, 6, 7, 2]])
                with torch.no_grad():
                    expected = adapted(input_ids=inputs).logits
                merge_adapter(str(base), final, merged, model_kind=kind, local_files_only=True)
                restored, _, _ = load_model_and_processor(str(merged), model_kind=kind, dtype='float32', local_files_only=True)
                restored.eval()
                with torch.no_grad():
                    actual = restored(input_ids=inputs).logits
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                with self.assertRaises(FileExistsError):
                    merge_adapter(str(base), final, merged, model_kind=kind, local_files_only=True)


if __name__ == '__main__':
    unittest.main()
