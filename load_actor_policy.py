from sb3_contrib.grpo.policies import ActorPolicy

PATH = "./logs/exp_16/ppo/CartPole-v1_4/best_model.zip"

ActorPolicy.load_from_actor_critic_policy(PATH)
