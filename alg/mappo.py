import torch
import numpy as np

from alg.actor_critic import Actor, Critic
from utils.utils import update_linear_schedule, huber_loss, mse_loss, check_array, get_gard_norm


class MAPPOPolicy:
    """
    Parameters:
        obs_space: (gym.Space) observation space
        cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
        action_space: (gym.Space) action space
    """
    def __init__(self, args, obs_dim, cent_obs_dim, act_dim, device=torch.device("cpu")):
        self.actor_lr = args.actor_lr  # 7e-4
        self.critic_lr = args.critic_lr  # 1e-3
        self.opti_eps = args.opti_eps  # a parameter used to set the epsilon value (eps) in the Adam optimizer
        self.weight_decay = args.weight_decay  # a parameter used to set the weight decay (also known as L2 regularization) for the Adam optimizer
        # Weight decay helps to prevent overfitting by penalizing large weights in the model. It adds a regularization term to the loss function, which encourages the optimizer to keep the weights small.

        self.obs_dim = obs_dim
        self.cent_obs_dim = cent_obs_dim
        self.act_dim = act_dim

        self.device = device
        self.actor = Actor(args, self.obs_dim, self.act_dim, self.device)
        self.critic = Critic(args, self.cent_obs_dim, self.device)

        # optimizers
        # self.actor.parameters() is a method that returns an iterator over all the parameters of the self.actor network. These parameters include all the weights and biases of the neural network that defines the actor model. The optimizer needs access to these parameters so that it can update them during training.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        Parameters:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        Parameters:
            cent_obs (np.ndarray): centralized input to the critic.
            obs (np.ndarray): local agent inputs to the actor.
            available_actions: (np.ndarray) denotes which actions are available to agent
                               (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        Returns:
            values: (torch.Tensor) value function predictions.
            actions: (torch.Tensor) actions to take.
            action_log_probs (torch.Tensor): log probabilities of the actions.
            action_probs (torch.Tensor): probabilities of the actions.
        """
        actions, action_log_probs, action_probs = self.actor(obs, available_actions, deterministic)
        values = self.critic(cent_obs)
        return values, actions, action_log_probs, action_probs

    def get_values(self, cent_obs) -> torch.Tensor:
        """
        Get value function predictions for the given centralized observations.
        Parameters:
            cent_obs (np.ndarray or torch.Tensor): centralized input to the critic.
        
        Returns:
            values (torch.Tensor): value function predictions.
        """
        values = self.critic(cent_obs)
        return values

    def evaluate_actions(self, cent_obs, obs, action, available_actions=None):
        """
        Evaluate the given actions.
        Parameters:
            cent_obs (np.ndarray or torch.Tensor): centralized input to the critic.
            obs (np.ndarray or torch.Tensor): local agent inputs to the actor.
            action (np.ndarray or torch.Tensor): actions to evaluate.
            available_actions (np.ndarray or torch.Tensor, optional): denotes which actions are available to agent
                                                                       (if None, all actions available)
        
        Returns:
            values (torch.Tensor): value function predictions.
            action_log_probs (torch.Tensor): log probabilities of the given actions.
            dist_entropy (torch.Tensor): entropy of the action distribution.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions)
        values = self.critic(cent_obs)
        return values, action_log_probs, dist_entropy

    def act(self, obs, available_actions=None, deterministic=False) -> torch.Tensor:
        """
        Compute actions using the given input.
        Parameters:
            obs (np.ndarray or torch.Tensor): local agent inputs to the actor.
            available_actions (np.ndarray or torch.Tensor, optional): denotes which actions are available to agent
                                                                       (if None, all actions available)
            deterministic (bool): whether the action should be mode of distribution or should be sampled.
        
        Returns:
            actions (torch.Tensor): actions to take.
        """
        actions, _, _ = self.actor(obs, available_actions, deterministic)
        return actions
    

class MAPPOTrainer():
    """
    Trainer class for MAPPO to update policies.
    Parameters:
        policy: (R_MAPPO_Policy) policy to update.
        device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, policy, device=torch.device("cpu")):
        # arguments
        self.clip_param = args.clip_param  # 0.2
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.huber_delta = args.huber_delta  # 10

        # switchers
        self._use_huber_loss = args.use_huber_loss
        self._use_clipped_value_loss = args.use_clipped_value_loss

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

    def calculate_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        Parameters:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values
        
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        
        value_loss = value_loss.mean()
        return value_loss
    
    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        Parameters:
            sample: (Tuple) contains data batch with which to update networks.
        
        """
        # Sample
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, \
        masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch = sample
        
        # convert to torch.Tensor if necessary
        old_action_log_probs_batch = check_array(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check_array(adv_targ).to(**self.tpdv)
        value_preds_batch = check_array(value_preds_batch).to(**self.tpdv)
        return_batch = check_array(return_batch).to(**self.tpdv)

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                obs_batch, actions_batch,
                                                                                available_actions_batch)
        # Actor Update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        # Surrogate loss
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        self.policy.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # Critic Update
        value_loss = self.calculate_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        Parameters:
            buffer: (SharedReplayBuffer) buffer containing training data.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
        
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info
    
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    