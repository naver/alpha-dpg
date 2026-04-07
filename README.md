# Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity
Germán Kruszewski, Pierre Erbacher, Jos Rozen, Marc Dymetman

[Paper](https://openreview.net/forum?id=pPWQUdhYSV) published at ICLR 2026.


## Setup

To setup the Lean environment,

1. Fist, follow the installation instructions of https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
2. Download the training and evaluation data used in "Rewarding the Unlikely":
  - https://github.com/AndreHe02/rewarding-unlikely-release/blob/master/data/mff-lwb-10k-seen.parquet
  - https://github.com/AndreHe02/rewarding-unlikely-release/blob/master/data/mff-lwb-unseen-200.parquet
3. Use [`examples/data_preprocess/lean.ipynb`](examples/data_preprocess/lean.ipynb) to pre-process prompts so that they match the COT prompt expected by DeepSeek-Prover-V1.5

## Training

You can find example training scripts at [`examples/lean/alpha-dpg.sh`](examples/lean/alpha-dpg.sh) and [`examples/lean/rl.sh`](examples/lean/rl.sh).

## Evaluation

To evaluate the models, follow these steps:

1. Convert the model to HF format:

```
python -m verl.model_merger merge --backend="fsdp" --local_dir="$CHECKPOINT_PATH/global_step_$STEPS/actor" --target_dir="$MODEL_PATH" --dtype="$DTYPE"
```

2. Generate samples from the evaluation set:
```
python -m verl.trainer.main_generation \
  model.path="$MODEL_PATH" \
  data.output_path="$OUTPUT_PATH/generations.parquet" \
  data.path="$DATA_PATH/mff-lwb-unseen-200.parquet" \
  data.prompt_key="prompt" \
  data.batch_size=100 \
  data.n_samples=256 \
  rollout.dtype="$DTYPE" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  rollout.temperature=1 \
  rollout.top_p="0.99" \
  rollout.top_k=-1 \
  rollout.prompt_length=1500 \
  rollout.response_length=1024 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.7 \
  rollout.max_num_batched_tokens=98304 \
  rollout.calculate_log_probs="true"
```

3. Evaluate the samples using the Lean environment:

```
python -m eval.eval \
  --generation_path="$OUTPUT_PATH/generations.parquet" \
  --output_path="$OUTPUT_PATH/results.parquet" \
  --split_id=0 \
  --total_split=1 \
  --nb_cpu=16
```
Sharding using multiple splits is recommended for faster and more robust evaluation.

## Citation

Please, cite our work using the following BibTeX entry:
```
@inproceedings{
  kruszewski2026whatever,
  title={Whatever Remains Must Be True: Filtering Drives Reasoning in {LLM}s, Shaping Diversity},
  author={Germ{\'a}n Kruszewski and Pierre ERBACHER and Jos Rozen and Marc Dymetman},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=pPWQUdhYSV}
}
```

Copyright (C) 2026-present NAVER Corporation.
