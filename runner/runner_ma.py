import time
import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.buffer import SharedReplayBuffer
from alg.mappo import MAPPOTrainer, MAPPOPolicy


class RunnerMA:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['env']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # interval
        #self.save_interval = self.all_args.save_interval
        #self.use_eval = self.all_args.use_eval
        #self.eval_interval = self.all_args.eval_interval
        
        self.save_dir = 'save_dir'
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_interval = self.all_args.log_interval

        share_observation_space = self.env.share_observation_space if self.use_centralized_V else self.env.observation_space

        # Policy
        obs_dim = self.env.observation_space.shape[0]
        cent_obs_dim = self.env.share_observation_space.shape[0]
        cent_obs_dim = cent_obs_dim
        act_dim = self.env.action_space.n

        self.policy = MAPPOPolicy(self.all_args, obs_dim, cent_obs_dim, act_dim)

        # Buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.env.observation_space,
                                         share_observation_space,
                                         self.env.action_space)
        
        # Trainer
        self.trainer = MAPPOTrainer(self.all_args, self.policy)
        
    def warmup(self, verbose=False):
        # reset env
        obs, share_obs, available_actions, info = self.env.reset()        
        
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

        if verbose:
            print('Warm up:')
            print(f'My init obs: {obs}')
            #print(f'My init all Obs in buffer: {self.buffer.obs}')
            print(f'Av actions: {self.buffer.available_actions[0]}')
            print('===')

    def run(self, verbose=False):
        #self.warmup(verbose=verbose)
        train_infos_history = []

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        # buffer = episode length = Maximum number of steps in an episode
        # in every episode can be several games, so will be several reset()
        # buffer = 1 episode
        for episode in range(episodes):
            # reset env
            self.warmup(verbose=verbose)

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            #print('episode #', episode)

            for step in range(self.episode_length):  # Max length for any episode
                #print('Max episode length:', self.episode_length)
                # (1) sample action
                if verbose:
                    print(f'Obs: {self.buffer.obs[step]}')
                    print(f'Av actions: {self.buffer.available_actions[step]}')

                values, actions, action_log_probs, action_probs, actions_env = self.collect(step)
                
                # (2) Make env step
                obs, share_obs, available_actions, rewards, done, info = self.env.step(actions_env)
                if verbose:
                    print(f'Chosen action: {actions_env}, reward: {rewards}')
                    print('Action probs:', action_probs)
                    print('Values:', values)
                    print('Done?', done)
                    if done:
                        print('========')

                data = obs, share_obs, rewards, done, values, actions, action_log_probs, available_actions

                # (3) insert data into buffer
                self.insert(data, verbose=verbose)
                
                if verbose:
                    episode_reward += rewards  # Accumulate reward for this episode
                    #print(f'Updated buffer: {self.buffer.obs}')

                if done:
                    obs, share_obs, available_actions, info = self.env.reset()  
                    self.buffer.share_obs[step+1] = share_obs.copy()
                    self.buffer.obs[step+1] = obs.copy()
                    self.buffer.available_actions[step+1] = available_actions.copy()          
        

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log info
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n{} / {} episodes, total number of timesteps: {}/{}, FPS {}. "
                      .format(
                          episode,
                          episodes,
                          total_num_steps,
                          self.num_env_steps,
                          int(total_num_steps / (end - start))
                      ))
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos_history.append(np.mean(self.buffer.rewards))
                print("average episode rewards is {} (av.reward: {} * ep_length {})"
                      .format(
                          train_infos["average_episode_rewards"],
                          np.mean(self.buffer.rewards),
                          self.episode_length
                          ))

        plt.figure()
        plt.plot(train_infos_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        values, actions, action_log_probs, action_probs = self.trainer.policy.get_actions(
            self.buffer.share_obs[step],
            self.buffer.obs[step],
            self.buffer.available_actions[step],
        )
        # convert to numpy
        values = np.array(values)
        actions = np.array(actions)
        action_log_probs = np.array(action_log_probs)

        # Action to put into env step() method
        actions_env: dict = self._convert_do_act_dict(actions)
        return values, actions, action_log_probs, action_probs, actions_env
    
    def insert(self, data, verbose=False):        
        obs, share_obs, rewards, done, values, actions, action_log_probs, available_actions = data
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                
        
        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
            available_actions=available_actions,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            self.buffer.share_obs[-1]
        )
        next_values = np.array(next_values)
        self.buffer.compute_returns(next_values)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        #self.buffer.after_update()
        return train_infos

    @torch.no_grad()
    def eval(self, num_episodes=10, episode_length=100, deterministic=False, verbose=False, verbose_render=False):
        """
        episode_length: (int) Maximum length of an episode
        """
        total_rewards = []
        for episode in range(num_episodes):
            if verbose:
                print(f'Episode #{episode+1}')
            #obs = self.env.reset()
            obs, share_obs, available_actions, info = self.env.reset()  
            done = False
            episode_reward = 0
            counter_step = 0
            while not done:
                if verbose:
                    print(f'Robot position: {self.env.robot_position}, Coin position: {self.env.coin_position}')
                # Get action from policy            
                values, actions, action_log_probs, action_probs = self.trainer.policy.get_actions(
                    share_obs,  # wrap in batch dimension
                    obs,
                    available_actions,
                    deterministic=deterministic
                )
                actions_env = self._convert_do_act_dict(actions)

                # Step environment
                obs, share_obs, available_actions, rewards, done, info = self.env.step(actions_env)                
                episode_reward += np.sum(rewards)
                counter_step += 1
                if counter_step == episode_length:
                    print('The maximum length of episode is reached.')
                    break
                    
                if verbose:
                    user_friendly_format = [[f"{num*100:.0f}%" for num in row] for row in action_probs.tolist()]
                    print(f'Action: {actions_env}, action probability: {user_friendly_format}')
                    
                    print('=======')
                if verbose_render:
                    self.env.render()

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}, Number of steps: {counter_step}")
            print('=======  EPISODE END  ================')
            print('')

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    def save(self):
        """
        Save policy's actor and critic networks
        """
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, model_dir):
        """
        Restore policy's networks from a saved model.
        """
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(model_dir) + '/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def _convert_do_act_dict(self, actions: torch.Tensor) -> dict:
        flattened_data = actions.flatten()
        action_dict = {agent_id: value.item() for agent_id, value in zip(self.env._agent_ids, flattened_data)}
        return action_dict


