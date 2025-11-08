---
language:
- zh
license: mit
size_categories:
- 1K<n<10K
source_datasets:
- gsm8k
task_categories:
- text2text-generation
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: question_zh-cn
    dtype: string
  - name: answer_only
    dtype: int64
  splits:
  - name: test
    num_bytes: 1020788
    num_examples: 1319
  - name: train
    num_bytes: 5664657
    num_examples: 7473
  download_size: 3988161
  dataset_size: 6685445
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: train
    path: data/train-*
tags:
- math-word-problems
---
