## Experiment Usage Guide

### 0. Preparatory Changes

Locally add your envs to RL-Zoo3's env_key_to_env_id dict in plot_from_file.py and the reference scores per env in RL-Zoo3's reference_scores in score_normalization.


### 1. Train Multiple Seeds

Run the following command to train multiple seeds for each algorithm and/or environment.  
**Note:** `num_timesteps` should be the same across the same experiment.

```bash
python experiment_rliable.py \
    --algo <algo_name> \
    --env <env_name> \
    -conf sb3_contrib/hyperparams/<algo_name>.yml \
    -n <num_timesteps> \
    --eval-freq <eval_freq> \
    -f logs/exp_<exp_id> \
    --seed <seed_id>
```

---

### 2. Generate Experiment PKL Data

Use this command to generate the experiment `.pkl` data:

```bash
python -m rl_zoo3.plots.all_plots.py \
    -a <algo_name(s)> \
    -e <env_name(s)> \
    -f logs \
    -o logs/exp_<exp_id>
```

---

### 3. Generate Rliable Plots

Create rliable plots with the following command:

```bash
python -m rl_zoo3.plots.plot_from_file \
    -i logs/exp_<exp_id>.pkl \
    -l <label(s)> \
    -r \
    -vs \
    -iqm \
    -b
```
- `-r`   : Enable rliable plots  
- `-vs`  : Enable probability of improvement plot  
- `-iqm` : Enable IQM sample efficiency plot  
- `-b`   : Enable boxplot  

