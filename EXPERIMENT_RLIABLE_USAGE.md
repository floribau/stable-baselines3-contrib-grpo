# Experiment Usage Guide with RL-Zoo3

## 0. Preparatory Changes

Locally add your environments to RL-Zoo3's `env_key_to_env_id` dictionary in [`rl_zoo3/plots/plot_from_file.py`](.venv/lib/python3.10/site-packages/rl_zoo3/plots/plot_from_file.py), and the reference scores per environment for reward normalization in RL-Zoo3's `reference_scores` in [`rl_zoo3/plots/score_normalization.py`](.venv/lib/python3.10/site-packages/rl_zoo3/plots/score_normalization.py).



## 1. Hyperparameter Tuning using Optuna

You can use Optuna (integrated into RL-Zoo3) to tune the hyperparameters of your custom algorithm.
Hyperparameter Tuning uses the same file as the Training, only with some additional args.

### Preparatory Changes for using Optuna
In order to use it, you need to first define a `sampler` and a `converter` function for the algorithm in [`rl_zoo3/hyperparams_opt.py`](.venv/lib/python3.10/site-packages/rl_zoo3/hyperparams_opt.py), and register the two in the `HYPERPARAMS_SAMPLER` and `HYPERPARAMS_CONVERTER` dicts.

Example converter function for GRPO:

```python
def convert_grpo_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    """
    Converter for GRPO hyperparams.

    :param sampled_params:
    :return: Converted hyperparams
    """
    hyperparams = sampled_params.copy()

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["group_size"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "tiny": [64],
        "small": [64, 64],
        "medium": [256, 256],
    }[net_arch]

    activation_fn_name = sampled_params["activation_fn"]
    del hyperparams["activation_fn"]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
        **hyperparams,
    }
```

Example sampler function for GRPO:
```python
def sample_grpo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for GRPO hyperparams.

    :param sampled_params:
    :return: Converted hyperparams
    """
    # From 2**2=4 to 2**8=256
    group_size_pow = trial.suggest_int("group_size_pow", 2, 8)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])

    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Display true values
    trial.set_user_attr("group_size", 2**group_size_pow)
    sampled_params = {
        "group_size_pow": group_size_pow,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    return convert_grpo_params(sampled_params)
```

### Command Lines for using Optuna

After having added the `converter` and `sampler` functions, run the following command to tune hyperparameters.
Optuna will tune the hyperparameters such that the found reward after `<num_timesteps>` env interactions is optimized.

```bash
python experiment_rliable_train.py \
    --algo <algo_name> \
    --env <env_name> \
    -conf sb3_contrib/hyperparams/<algo_name>.yml \
    -n <num_timesteps> \
    -optimize \
    --n-trials <num_trials>> \
    --optimization-log-path ./optuna/<algo_name>/<env_name>/opt_<run_id>
```

- `-n`         : The number of timesteps per trial. Optuna optimizes the achieved reward after `<num_timesteps>`env interactions
- `--n-trials` : The number of trials during Tuning. The higher the value, the more hyperparameter combinations can be tried by Optuna.



## 2. Collect Training Data across multiple runs

Run the following command to perform one training run. This should be run multiple times to collect enough data.
Make sure your custom algorithms are registered in `ALGOS` in [`experiment_rliable_train.py`](experiment_rliable_train.py).

**Note:** `num_timesteps` should be the same across all runs of the same experiment.

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
- `--eval-freq`: Make sure to have at least 5 eval points.

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
    for ALGO_NAME in ppo process-grpo outcome-grpo; do
        python experiment_rliable_train.py --algo $ALGO_NAME --env $ENV -conf sb3_contrib/hyperparams/$ALGO_NAME.yml -n $NUM_TIMESTEPS --eval-freq $EVAL_FREQ -f $LOG_PATH --seed $SEED
    done
done
```

---

## 3. Aggregate Experiment PKL Data

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

## 4. Generate Rliable Plots

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

