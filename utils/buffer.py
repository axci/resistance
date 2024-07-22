import numpy as np
import torch
from utils.utils import get_shape_from_act_space, get_shape_from_obs_space


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    Parameters:
        args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
            episode_length: (int) Maximum number of steps in an episode
            n_rollout_threads: (int) Number of parallel environments or threads used for collecting rollouts.
            gamma: (float) discount factor
        
        num_agents: (int) number of agents in the env.
        obs_space: (gym.Space) observation space of agents.
        cent_obs_space: (gym.Space) centralized observation space of agents.
        act_space: (gym.Space) action space for agents.
    """
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        # args
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads  # specifies the number of parallel threads or environments used for collecting rollouts
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args._use_gae
        
        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        act_shape = get_shape_from_act_space(act_space)  # 1

        self.obs = np.zeros(
            (self.episode_length + 1,
            self.n_rollout_threads,
            self.num_agents,
            obs_shape
            ), dtype=np.float32)
        self.share_obs = np.zeros(
            (self.episode_length + 1,
            self.n_rollout_threads,
            self.num_agents,
            share_obs_shape
            ), dtype=np.float32)
        self.value_preds = np.zeros(
            (self.episode_length + 1, 
             self.n_rollout_threads,
             self.num_agents,
             1
             ), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, 
             self.n_rollout_threads,
             self.num_agents,
             1
             ), dtype=np.float32)
        self.available_actions = np.ones((
            self.episode_length + 1,
            self.n_rollout_threads,
            self.num_agents,
            act_space.n            
        ), dtype=np.float32)
        self.actions = np.zeros((
            self.episode_length,
            self.n_rollout_threads,
            num_agents,
            act_shape
        ), dtype=np.float32)
        self.action_log_probs = np.zeros((
            self.episode_length,
            self.n_rollout_threads,
            num_agents,
            act_shape
        ), dtype=np.float32)
        self.rewards = np.zeros((
            self.episode_length,
            self.n_rollout_threads,
            self.num_agents,
            1
        ), dtype=np.float32)
        
        # Initialize the buffer for masks (1 if ongoing, 0 if terminated)
        # self.masks = np.ones((  
        #     self.episode_length + 1, 
        #     self.n_rollout_threads, 
        #     num_agents, 
        #     1
        # ), dtype=np.float32)

        self.step = 0

    def insert(self, share_obs: np.ndarray, obs: np.ndarray, 
               actions: np.ndarray, action_log_probs: np.ndarray, 
               value_preds: np.ndarray, rewards: np.ndarray, 
               #masks: np.ndarray, 
               available_actions: np.ndarray=None):
        """
        Insert data into the buffer.
        Parameters:
            obs (np.ndarray) local agent observations.
            actions:(np.ndarray) actions taken by agents.
            action_log_probs:(np.ndarray) log probs of actions taken by agents
            value_preds: (np.ndarray) value function prediction at each step.
            rewards: (np.ndarray) reward collected at each step.
            masks: (np.ndarray) denotes whether the environment has terminated or not.
            available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """                             
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        #self.masks[self.step + 1] = masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        # Increase step until reach episode_length, then 0
        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, actions, action_log_probs,
                     value_preds, rewards, 
                     #masks, 
                     available_actions=None):
        """
        Insert data into the buffer. For turn based.
        Parameters:
            obs: (np.ndarray) local agent observations.
            actions:(np.ndarray) actions taken by agents.
            action_log_probs:(np.ndarray) log probs of actions taken by agents
            value_preds: (np.ndarray) value function prediction at each step.
            rewards: (np.ndarray) reward collected at each step.
            masks: (np.ndarray) denotes whether the environment has terminated or not.
            available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        #self.masks[self.step + 1] = masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        #self.masks[0] = self.masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
            next_value: (np.ndarray) value predictions for the step after the last episode step.
        """
        if self._use_gae:
            self.value_preds[-1] = next_value  # Assign the value prediction for the step after the last episode step
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                # Compute the temporal difference error
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] - self.value_preds[step]
                # Accumulate the GAE
                gae = delta + self.gamma * self.gae_lambda * gae
                # Compute the return
                self.returns[step] = gae + self.value_preds[step]
        else:  # Assign the value prediction for the step after the last episode step
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                # Compute the discounted return
                self.returns[step] = self.returns[step + 1] * self.gamma + self.rewards[step]
    
    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        Parameters:
            advantages: (np.ndarray) advantage estimates.
            num_mini_batch: (int) number of minibatches to split the batch into.
            mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()  # random permutation. If batch_size = 5, the result can be array([3, 2, 4, 1, 0])
        
        # Sampler: If batch_size = 5, num_mini_batch = 2, then mini_batch_size = 5 // 2 = 2
        #          The result might be: [array([4, 1]), array([0, 3])]  (from above rand)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        
        """
        rows = array([[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4]])
        
        cols =array([[0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4]])
        """      
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        #masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        #masks = masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            #masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, \
                  old_action_log_probs_batch, adv_targ, available_actions_batch

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        #masks = self.masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            #masks_batch = masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, \
                  old_action_log_probs_batch, adv_targ, available_actions_batch