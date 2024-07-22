# Example configuration
class Args:
    def __init__(self):
        self.env_name = 'Resistance Env'
        self.algorithm_name = 'MAPPO'
        self.experiment_name = 'Res Experience'
        self.use_centralized_V = True
        self.num_env_steps = 10_000_000
        self.episode_length = 10_000
        self.n_rollout_threads = 1
        self.use_linear_lr_decay = False
        self.hidden_size = 64
        self.gamma = .99
        self.gae_lambda = .95
        self._use_gae = True
        self.actor_lr = 7e-4
        self.critic_lr = 1e-3
        self.opti_eps=1e-5
        self.weight_decay=0
        self.gamma=0.99
        self.clip_param=0.2
        self.value_loss_coef=1
        self.entropy_coef=0.01
        self.ppo_epoch=10
        self.num_mini_batch=1
        self.huber_delta=10.0
        self.use_huber_loss=True
        self.use_clipped_value_loss=True
        self.gain = 0.01
        self.use_orthogonal = True
        self.use_feature_normalization = True
        self.use_ReLU = True
        self.layer_N = 1
        self.log_interval = 100_000  # log every x steps
        