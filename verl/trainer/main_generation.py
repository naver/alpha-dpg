# alpha-dpg
# Modified by Copyright (C) 2026 Naver Corporation. All rights reserved.

# Original work
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or im the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
from pprint import pprint
import shutil

import hydra
import mlflow
import numpy as np
import pandas as pd
import ray
import torch
import random
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                     RayWorkerGroup)
from verl.utils import hf_tokenizer
from verl.utils.debug import simple_timer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_USAGE_STATS_ENABLED"] = "0"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


def set_seed(seed: int):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in certain torch operations if needed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Set the random seed from config
    seed = config.get("trainer", {}).get("seed", 1)
    set_seed(seed)

    # mlflow.set_experiment(config.trainer.project_name)
    # mlflow.start_run(run_name=config.trainer.run_name)
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    assert config.rollout.calculate_log_probs, "rollout.config_log_probs must be true because we now save the score of the generated sequences."

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    # Define the slice start point
    start = config.data.slice_start if config.data.slice_start is not None else 0
    # Define the slice end point
    end = (start + config.data.crop_size) if config.data.crop_size is not None else None
    # Slice the dataset
    dataset = dataset[start:end]
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    # add system prompt
    for i in range(len(chat_lst)):
        chat_lst[i].append({"role": "system", "content": "You are an expert in mathematics and Lean 4."})


    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    log_probs_lst = [[] for _ in range(config.data.n_samples)]
    tokenized_responses_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            timings = {}
            with simple_timer("gen", timings):
                output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            # Get tensors from the output batch
            rollout_log_probs = output.batch["rollout_log_probs"]  # Shape: (bs, response_length)
            responses = output.batch["responses"]  # Shape: (bs, response_length)

            # 1. Create a mask for the valid (non-padding) tokens within the response.
            response_mask = get_response_mask(
                responses, eos_token=tokenizer.eos_token_id, dtype=output.batch["attention_mask"].dtype
            )

            # 2. Calculate the log probs by applying the mask.
            log_probs = (rollout_log_probs * response_mask).sum(dim=-1)
            num_samples = output.batch["responses"].shape[0]
            response_length = output.batch["responses"].shape[-1]

            time_per_sample_s = timings["gen"] / num_samples
            time_per_token_ms = time_per_sample_s / response_length * 1000
            step_idx = batch_idx * config.data.n_samples + n_sample
            # mlflow.log_metrics(
            #     {
            #         "time_per_sample_s": time_per_sample_s,
            #         "time_per_token_ms": time_per_token_ms,
            #     },
            #     step_idx,
            # )

            valid_ids_list = [
                item.batch["responses"][: item.batch["attention_mask"][item.batch["prompts"].shape[-1] :].sum()]
                for item in output
            ]
            output_texts = tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)

            # Convert the list of tensors to a list of lists
            valid_ids_as_list = [ids.tolist() for ids in valid_ids_list]
            tokenized_responses_lst[n_sample].extend(valid_ids_as_list)

            output_lst[n_sample].extend(output_texts)
            log_probs_lst[n_sample].extend(log_probs.tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    log_probs_lst = np.array(log_probs_lst)
    log_probs_lst = np.transpose(log_probs_lst, axes=(1, 0)).tolist()

    tokenized_responses_lst = np.array(tokenized_responses_lst, dtype=object)
    tokenized_responses_lst = np.transpose(tokenized_responses_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst
    dataset["log_probs"] = log_probs_lst
    dataset["tokenized_responses"] = tokenized_responses_lst

    output_path = config.data.output_path
    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)

    backup_path = output_path + ".bak"
    try:
        if os.path.exists(output_path):
            print(f"Output file found at '{output_path}'. Creating backup at '{backup_path}'.")
            # Create a backup of the existing file
            shutil.copy2(output_path, backup_path)
            print(f"Backup created successfully.")

            # OVERWRITE the original file with the new dataset
            print(f"Writing new data, overwriting original file at '{output_path}'.")
            dataset.to_parquet(output_path, index=False)
            print(f"Successfully saved new data to '{output_path}'.")
        else:
            # If no file exists, create a new one
            print(f"No existing file found. Writing new data to '{output_path}'.")
            dataset.to_parquet(output_path, index=False)
            print(f"Successfully saved new data to '{output_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


if __name__ == "__main__":
    main()