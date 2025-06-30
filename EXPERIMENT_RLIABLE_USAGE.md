## Experiment Usage Guide

### 0. Preparatory Changes

Locally add your environments to RL-Zoo3's `env_key_to_env_id` dictionary in [`rl_zoo3/plots/plot_from_file.py`](.venv/lib/python3.10/site-packages/rl_zoo3/plots/plot_from_file.py), and the reference scores per environment for reward normalization in RL-Zoo3's `reference_scores` in [`rl_zoo3/plots/score_normalization.py`](.venv/lib/python3.10/site-packages/rl_zoo3/plots/score_normalization.py).


### 1. Train Multiple Seeds

Run the following command to train multiple seeds for each algorithm and/or environment.  
**Note:** `num_timesteps` should be the same across the same experiment.

```bash
python experiment_rliable_train.py \
    --algo <algo_name> \
    --env <env_name> \
    -conf sb3_contrib/hyperparams/<algo_name>.yml \
    -n <num_timesteps> \
    --eval-freq <eval_freq> \
    -f logs/exp_<exp_id> \
    --seed <seed_id>
```
- eval-freq: Make sure to have at least 5 eval points.

Alternatively, you can run the following comman line once for each env
```bash
ENV=<env_name>
NUM_TIMESTEPS=<num_timesteps>
EVAL_FREQ=<eval_freq>
LOG_PATH=logs/exp_<exp_id>

for SEED in 0 1 2 3 4; do
    for ALGO_NAME in <algo_names>; do
        python experiment_rliable_train.py --algo $ALGO_NAME --env $ENV -conf sb3_contrib/hyperparams/$ALGO_NAME.yml -n $NUM_TIMESTEPS --eval-freq $EVAL_FREQ -f $LOG_PATH --seed $SEED
    done
done
```

Prefilled:
```bash
ENV=CartPole-v1
NUM_TIMESTEPS=50000
EVAL_FREQ=1000
LOG_PATH=logs/exp_<exp_id>

for SEED in 0 1 2 3; do
    for ALGO_NAME in ppo process-grpo; do
        python experiment_rliable_train.py --algo $ALGO_NAME --env $ENV -conf sb3_contrib/hyperparams/$ALGO_NAME.yml -n $NUM_TIMESTEPS --eval-freq $EVAL_FREQ -f $LOG_PATH --seed $SEED
    done
done
```

---

### 2. Generate Experiment PKL Data

Use this command to generate the experiment `.pkl` data.
It will first truncate any excess evaluation steps to the minimum per experiment id, algorithm, and env. It will then call [`rl_zoo3/plots/all_plots.py`](.venv/lib/python3.10/site-packages/rl_zoo3/plots/all_plots.py)

```bash
python experiment_rliable_all_plots.py \
    -a <algo_name(s)> \
    -e <env_name(s)> \
    -f logs/exp_<exp_id> \
    -o logs/exp_<exp_id>/all_plots
```

---

### 3. Generate Rliable Plots

Create rliable plots with the following command:

```bash
python -m rl_zoo3.plots.plot_from_file \
    -i logs/exp_<exp_id>/all_plots.pkl \
    -l <label(s)> \
    -r \
    -vs \
    -iqm \
    -b \
    -o logs/exp_<exp_id>/plot_from_file
```
- `-r`   : Enable rliable plots  
- `-vs`  : Enable probability of improvement plot  
- `-iqm` : Enable IQM sample efficiency plot  
- `-b`   : Enable boxplot  

