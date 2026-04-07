# alpha-dpg
# Copyright (C) 2026 Naver Corporation. All rights reserved.

import os
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import datasets
from lean.verifier import verify_with_deepseek_verifier
import math
import numpy as np
import json
from tqdm import tqdm
import concurrent.futures
import re
from functools import partial
import time


def pass_at_k(n, c, k):
    """
    Calculates the pass@k metric.
    n: total number of samples.
    c: number of correct samples.
    k: the 'k' in pass@k.
    """
    # Handle the edge case where there are no correct samples.
    if c == 0:
        return 0.0
    # If the number of incorrect samples is less than k, at least one correct one must be chosen.
    if n - c < k:
        return 1.0
    # General formula for pass@k
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)




import datasets


def eval(x, nb_cpu):
    completions = x["responses"]
    data_sources = x["prompt"]
    prompt = x['reward_model']['ground_truth']
    start_time = time.time()

    
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=nb_cpu) as executor:
       
        future_to_completion = {executor.submit(verify_with_deepseek_verifier, '', k, prompt, verbose=True): k for k in completions}
        
        
        verification_pbar = tqdm(concurrent.futures.as_completed(future_to_completion), total=len(completions), desc="Verifying completions", leave=False)
        
        for future in verification_pbar:
            try:
                result = future.result()
                res.append(result)
            except Exception as exc:
                print(f'\nA verification task generated an exception: {exc}')
                res.append(False) 

    duration = time.time() - start_time
    print(f"Processed entry in {duration:.2f}s")
    return {"results" : res}


def main(generation_path,output_path, split_id, total_split, nb_cpu):
    
    print("Loading and processing dataset...")

    ds = datasets.load_dataset("parquet", data_files= generation_path)['train']
    print(f"Obtaining shard {split_id} of {total_split}...")
    ds = ds.shard(num_shards=total_split, index=split_id)
    print(f"Processing {len(ds)} entries...")
    eval_partial = partial(eval, nb_cpu=nb_cpu)
    ds = ds.map(eval_partial, load_from_cache_file=False)
    print(f"Saving {len(ds)} entries...")
    ds.to_parquet(os.path.join(output_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM evaluation with parallel verification.")
    parser.add_argument("--generation_path", type=str, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--output_path", type=str, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--split_id", type=int, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--total_split", type=int, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--nb_cpu", type=int, required=True, help="Hugging Face model name or path.")

    args = parser.parse_args()

    main(
        generation_path=args.generation_path,
        output_path=args.output_path,
        split_id=args.split_id,
        total_split=args.total_split,
        nb_cpu=args.nb_cpu
    )
