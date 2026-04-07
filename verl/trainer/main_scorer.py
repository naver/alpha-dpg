# alpha-dpg
# Copyright (C) 2026 Naver Corporation. All rights reserved.

import os
from collections import defaultdict

import hydra
import pandas as pd
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

# Assuming 'verl' is an installed package in your environment
from verl import DataProto
from verl.utils import hf_tokenizer

# Import the new utility functions
from verl.trainer.ppo.reward import get_cached_reward_fn, default_compute_score


@ray.remote
class RewardWorker:
    """A simplified Ray actor that directly calls the scoring function."""
    def __init__(self, config_str: str):
        self.score_fn = get_cached_reward_fn(config_str) or default_compute_score

    def compute_scores_from_batch(self, batch_df: pd.DataFrame) -> Tuple[List[float], Dict[str, List[Any]]]:
        """Processes a batch of data by iterating over rows and calling the score function."""
        scores = []
        extra_info_agg = defaultdict(list)

        for _, row in batch_df.iterrows():
            # Construct the arguments for the score function directly from the DataFrame row
            score_args = {
                "solution_str": row["response"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "data_source": row.get("data_source"),
            }
            
            # Call the score function directly
            result = self.score_fn(**score_args)
            
            # Process the result (which could be a dict or a single float)
            if isinstance(result, dict):
                score = result.pop("score", 0.0)
                for key, value in result.items():
                    extra_info_agg[key].append(value)
            else:
                score = float(result)

            scores.append(score)
            
        return scores, dict(extra_info_agg)

def run_scoring(config: DictConfig) -> None:
    """Main function to orchestrate the scoring of generated sequences."""
    print("Configuration:\n", OmegaConf.to_yaml(config))

    df = pd.read_parquet(config.data.input_path)
    df['original_index'] = df.index
    df_long = df.explode(config.data.responses_key, ignore_index=False).rename(columns={config.data.responses_key: 'response'})
    print(f"Loaded dataset with {len(df)} prompts and {len(df_long)} total responses to score.")

    all_scores: List[float] = []
    all_extra_info = defaultdict(list)
    config_str = OmegaConf.to_yaml(config)

    if config.scoring.launch_reward_fn_async:
        print("\n🚀 Starting ASYNCHRONOUS scoring...")
        if not ray.is_initialized():
            ray.init(num_cpus=config.ray_init.num_cpus)
        
        workers = [RewardWorker.remote(config_str) for _ in range(config.scoring.num_workers)]
        
        results_futures = []
        for i in range(0, len(df_long), config.scoring.batch_size):
            batch_df = df_long.iloc[i:i + config.scoring.batch_size]
            worker = workers[i // config.scoring.batch_size % len(workers)]
            results_futures.append(worker.compute_scores_from_batch.remote(batch_df))

        for future in tqdm(results_futures, desc="Scoring Batches (Async)"):
            scores, extra_info = ray.get(future)
            all_scores.extend(scores)
            for key, val_list in extra_info.items():
                all_extra_info[key].extend(val_list)
    else:
        print("\n🐢 Starting SYNCHRONOUS scoring...")
        score_fn = get_cached_reward_fn(config_str) or default_compute_score
        
        for _, row in tqdm(df_long.iterrows(), total=len(df_long), desc="Scoring Samples (Sync)"):
            score_args = {
                "solution_str": row["response"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "data_source": row.get("data_source", "unknown"),
                "extra_info": {}
            }
            result = score_fn(**score_args)
            
            if isinstance(result, dict):
                score = result.pop("score", 0.0)
                for key, value in result.items():
                    all_extra_info[key].append(value)
            else:
                score = float(result)
            
            all_scores.append(score)

    # --- Aggregate Results and Save (This part remains the same) ---
    print("\nAggregating scores back into the original format...")
    df_long['reward'] = all_scores
    for key, values in all_extra_info.items():
        df_long[key] = values

    final_columns = ['reward'] + list(all_extra_info.keys())
    for col in final_columns:
        grouped_col = df_long.groupby('original_index')[col].apply(list)
        df[col] = grouped_col

    if 'reward_model' not in df.columns and 'reward_model' in df_long.columns:
         df = df.merge(df_long[['original_index', 'reward_model']].drop_duplicates(), on='original_index', how='left')

    df = df.drop(columns=['original_index'])

    os.makedirs(os.path.dirname(config.data.output_path), exist_ok=True)
    df.to_parquet(config.data.output_path, index=False)
    print(f"✅ Successfully saved scored dataset to: {config.data.output_path}")
    print("\nFinal DataFrame preview:")
    print(df.head())


@hydra.main(config_path="config", config_name="scoring", version_base=None)
def main(config: DictConfig) -> None:
    run_scoring(config)


if __name__ == "__main__":
    main()