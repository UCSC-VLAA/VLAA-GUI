#!/usr/bin/env bash

python run_multienv_vlaa.py \
    --config_path "configs/your_config.toml" \
    --pricing_config_path "path_to_pricing.toml" \
    --result_dir "results/your_results_dir" \
    --num_envs 10 \
    --test_config_base_dir "evaluation_examples" \
    --test_all_meta_path "evaluation_examples/test_nogdrive.json" \
    --domain "all" \
    --max_steps 100 \
    --client_password "osworld-public-evaluation" \
    --max_trajectory_length 8