"""Check notebook Trainer compatibility with a tiny local DeBERTa configuration.

No pretrained weights or course data are downloaded. Synthetic examples and
random model weights test training, evaluation, checkpoint export and reload.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForSequenceClassification, Trainer,
    TrainingArguments, default_data_collator,
)


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/claim_verification.ipynb').read_text(encoding='utf8'))
    definitions = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions[node.name] = node
    selected = ('class_weights', 'compute_metrics', 'WeightedTrainer', 'make_training_args', 'make_weighted_trainer', 'CrossEncoderReranker')
    namespace = dict(globals(), CE_MODEL='local-synthetic-model', CE_MAX_LEN=16, CE_BATCH_SIZE=2)
    exec('from __future__ import annotations\n' + '\n\n'.join(
        ast.unparse(definitions[name]) for name in selected), namespace)
    # DeBERTa-v3 uses the DebertaV2 encoder implementation in Transformers.
    config = DebertaV2Config(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        num_labels=4, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    torch.manual_seed(42)
    model = DebertaV2ForSequenceClassification(config)
    data = Dataset.from_dict({
        'input_ids': [[1, 4+i, 2, 0] for i in range(4)],
        'attention_mask': [[1, 1, 1, 0]]*4,
        'labels': [0, 1, 2, 3],
    })
    with tempfile.TemporaryDirectory(prefix='claim-trainer-check-') as temporary:
        directory = Path(temporary)
        backend = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'solar': 2,
            'panels': 3, 'electricity': 4, 'wind': 5}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            pad_token='[PAD]', unk_token='[UNK]', model_max_length=16)
        args = namespace['make_training_args']({
            'output_dir': str(directory/'train'), 'use_cpu': True, 'max_steps': 1,
            'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 4,
            'save_strategy': 'no', 'eval_strategy': 'no', 'report_to': 'none',
            'disable_tqdm': True, 'dataloader_pin_memory': False,
        })
        trainer = namespace['make_weighted_trainer'](
            model=model, args=args, train_dataset=data, eval_dataset=data,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            compute_metrics=namespace['compute_metrics'],
            class_weight=namespace['class_weights']([0,1,2,3], 4),
        )
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        metrics = trainer.evaluate()
        assert np.isfinite(metrics['eval_loss'])
        predictions = trainer.predict(data).predictions
        assert predictions.shape == (4, 4)
        trainer.save_model(str(directory/'saved'))
        restored = DebertaV2ForSequenceClassification.from_pretrained(directory/'saved')
        restored.eval()
        model.eval()
        batch = {k: torch.tensor(data[k]) for k in ('input_ids','attention_mask')}
        with torch.no_grad():
            expected = model(**batch).logits
            actual = restored(**batch).logits
        torch.testing.assert_close(actual, expected)
        assert trainer.args.eval_strategy.value == 'no'
        # MiniLM's reranking interface follows BertForSequenceClassification.
        # A tiny local checkpoint verifies the original loader and batching code.
        reranker_dir = directory/'reranker'
        rerank_config = BertConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, num_labels=1)
        BertForSequenceClassification(rerank_config).save_pretrained(reranker_dir)
        tokenizer.save_pretrained(reranker_dir)
        reranker = namespace['CrossEncoderReranker'](model_name=reranker_dir, fp16=False)
        scores = reranker.score([('solar panels', 'electricity'), ('wind', 'electricity')])
        assert scores.shape == (2,) and np.isfinite(scores).all()
        assert reranker.score([]).shape == (0,)
    print('PASS: WeightedTrainer construction, weighted loss, one CPU optimizer step, evaluation, prediction, save and reload')
    print(f'PyTorch {torch.__version__}; Transformers {transformers.__version__}')
    print('PASS: tokenizer API adaptation and local cross-encoder loading/batched scoring')
    print('Synthetic compatibility check only; no pretrained model or course data used.')


if __name__ == '__main__':
    main()
