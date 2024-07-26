import time
import torch
import os
from typing import Tuple
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from env.res_env_all import ResistanceFinalEnv
from env.naive_agents import NaiveEvilPolicy
from runner.game import simulate_game
from utils.buffer import SharedReplayBuffer
from alg.mappo import MAPPOTrainer, MAPPOPolicy



class RunnerMA_Both:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['env']
        self.device = config['device']
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length


        self.use_centralized_V = self.all_args.use_centralized_V
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        self.num_agents_evil = self.env.num_evil
        self.num_agents_good = self.env.num_good
        # interval
        #self.save_interval = self.all_args.save_interval
        #self.use_eval = self.all_args.use_eval
        #self.eval_interval = self.all_args.eval_interval
        
        self.save_dir = 'models'
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_interval = self.all_args.log_interval

        share_observation_space = self.env.share_observation_space if self.use_centralized_V else self.env.observation_space
        share_observation_space_good = self.env.share_observation_space_good if self.use_centralized_V else self.env.observation_space_good

        ##### EVIL SETUP
        # Evil Policy
        obs_dim_evil = self.env.observation_space.shape[0]
        cent_obs_dim_evil = self.env.share_observation_space.shape[0]
        act_dim_evil = self.env.action_space.n
        self.policy_evil = MAPPOPolicy(self.all_args, obs_dim_evil, cent_obs_dim_evil, act_dim_evil)

        # Evil Buffer
        self.buffer_evil = SharedReplayBuffer(self.all_args,
                                         self.num_agents_evil,
                                         self.env.observation_space,
                                         share_observation_space,
                                         self.env.action_space)
        
        # Evil Trainer
        self.trainer_evil = MAPPOTrainer(self.all_args, self.policy_evil)

        ##### GOOD SETUP
        # good Policy
        obs_dim_good = self.env.observation_space_good.shape[0]
        cent_obs_dim_good = self.env.share_observation_space_good.shape[0]
        act_dim_good = self.env.action_space_good.n
        self.policy_good = MAPPOPolicy(self.all_args, obs_dim_good, cent_obs_dim_good, act_dim_good)

        # good Buffer
        self.buffer_good = SharedReplayBuffer(self.all_args,
                                         self.num_agents_good,
                                         self.env.observation_space_good,
                                         share_observation_space_good,
                                         self.env.action_space_good)
        
        # good Trainer
        self.trainer_good = MAPPOTrainer(self.all_args, self.policy_good)
        
    def warmup(self, verbose=False):
        # reset env
        obs, share_obs, available_actions, info = self.env.reset()
        obs_evil, obs_good = self._convert_dict_to_two_2darrays(obs)
        share_obs_evil, share_obs_good = self._convert_dict_to_two_2darrays(share_obs)
        available_actions_evil, available_actions_good = self._convert_dict_to_two_2darrays(available_actions)
        # Evil buffer
        self.buffer_evil.share_obs[0] = share_obs_evil.copy()
        self.buffer_evil.obs[0] = obs_evil.copy()
        self.buffer_evil.available_actions[0] = available_actions_evil.copy()
        # Good buffer
        self.buffer_good.share_obs[0] = share_obs_good.copy()
        self.buffer_good.obs[0] = obs_good.copy()
        self.buffer_good.available_actions[0] = available_actions_good.copy()

        if verbose:
            print('Warm up:')
            print(f'My init obs: {obs}')
            print('===')

    def run(self, verbose=False):
        print('Start training.')
        print(f"Config. Hidden sizes: {self.all_args.hidden_sizes_list}")

        #self.warmup(verbose=verbose)
        train_infos_evil_history = []
        train_infos_good_history = []

        start = time.time()
        episodes = self.num_env_steps // self.episode_length // self.n_rollout_threads
        
        # buffer = episode length = Maximum number of steps in an episode
        # in every episode can be several games, so will be several reset()
        # buffer = 1 episode
        for episode in range(episodes):
            self.warmup(verbose=verbose) # reset env

            if self.use_linear_lr_decay:
                self.trainer_evil.policy.lr_decay(episode, episodes)
                self.trainer_good.policy.lr_decay(episode, episodes)

            for step in tqdm(range(self.episode_length)):  # Max length for any episode
                # (1) sample action               
                values_evil, actions_evil, action_log_probs_evil, action_probs_evil, \
                    values_good, actions_good, action_log_probs_good, action_probs_good, actions_env = self.collect(step)
                
                # (2) Make env step
                obs, share_obs, available_actions, rewards, done, info = self.env.step(actions_env)
                if verbose:
                    print(f'Chosen action: {actions_env}, reward: {rewards}')
                    print('Done?', done)
                    if done:
                        print('========')

                data = obs, share_obs, available_actions, rewards, \
                    values_evil, actions_evil, action_log_probs_evil, action_probs_evil, \
                    values_good, actions_good, action_log_probs_good, action_probs_good

                # (3) insert data into buffer
                self.insert(data)              
        
                if done:
                    obs, share_obs, available_actions, info = self.env.reset()

                    obs_evil, obs_good = self._convert_dict_to_two_2darrays(obs)
                    share_obs_evil, share_obs_good = self._convert_dict_to_two_2darrays(share_obs)
                    available_actions_evil, available_actions_good = self._convert_dict_to_two_2darrays(available_actions)
                    # Evil buffer
                    self.buffer_evil.share_obs[step+1] = share_obs_evil.copy()
                    self.buffer_evil.obs[step+1] = obs_evil.copy()
                    self.buffer_evil.available_actions[step+1] = available_actions_evil.copy()
                    # Good buffer
                    self.buffer_good.share_obs[step+1] = share_obs_good.copy()
                    self.buffer_good.obs[step+1] = obs_good.copy()
                    self.buffer_good.available_actions[step+1] = available_actions_good.copy()                       
        
            # compute return and update network
            self.compute()
            train_infos_evil, train_infos_good = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # Save
            if episode % 10 == 0:
                print('Policy is saved.')
                self.save('temp')

            # if np.mean(self.buffer_good.rewards) > -0.1:
            #     good_reward_av = np.mean(self.buffer_good.rewards)
            #     self.save(f'256_2_good_rew_{good_reward_av}')

            # log info
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n{} / {} episodes, total number of timesteps: {}/{}, Time: {} minutes. "
                      .format(
                          episode,
                          episodes,
                          total_num_steps,
                          self.num_env_steps,
                          int( (end - start)/60 ) 
                      ))
                train_infos_evil_history.append(np.mean(self.buffer_evil.rewards))
                train_infos_good_history.append(np.mean(self.buffer_good.rewards))

                print("Average episode rewards: 😈 {}, 😇 {}."
                      .format(
                          np.mean(self.buffer_evil.rewards),
                          np.mean(self.buffer_good.rewards),
                          ))
                
            # evaluation
            if total_num_steps % (self.log_interval*10) == 0:
                self.eval_good(100)

        plt.figure()
        plt.plot(train_infos_evil_history)
        plt.plot(train_infos_good_history)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Progress')
        plt.show()

    @torch.no_grad()
    def collect(self, step):
        self.trainer_evil.prep_rollout()
        self.trainer_good.prep_rollout()
        
        # GET ACTIONS: Evil
        values_evil, actions_evil, action_log_probs_evil, action_probs_evil = self.trainer_evil.policy.get_actions(
            self.buffer_evil.share_obs[step],
            self.buffer_evil.obs[step],
            self.buffer_evil.available_actions[step],
        )
        # convert to numpy
        values_evil = np.array(values_evil)
        actions_evil = np.array(actions_evil)
        action_log_probs_evil = np.array(action_log_probs_evil)

        # GET ACTIONS: Good
        values_good, actions_good, action_log_probs_good, action_probs_good = self.trainer_good.policy.get_actions(
            self.buffer_good.share_obs[step],
            self.buffer_good.obs[step],
            self.buffer_good.available_actions[step],
        )
        # convert to numpy
        values_good = np.array(values_good)
        actions_good = np.array(actions_good)
        action_log_probs_good = np.array(action_log_probs_good)

        # Action to put into env step() method
        actions_env: dict = self._convert_to_act_dict(actions_evil, actions_good)
        return values_evil, actions_evil, action_log_probs_evil, action_probs_evil, \
               values_good, actions_good, action_log_probs_good, action_probs_good, actions_env
    
    def insert(self, data):        
        obs, share_obs, available_actions, rewards, \
                    values_evil, actions_evil, action_log_probs_evil, action_probs_evil, \
                    values_good, actions_good, action_log_probs_good, action_probs_good = data

        # from dict to separate nd.arrays 
        obs_evil, obs_good               = self._convert_dict_to_two_2darrays(obs)
        share_obs_evil, share_obs_good   = self._convert_dict_to_two_2darrays(share_obs)
        av_actions_evil, av_actions_good = self._convert_dict_to_two_2darrays(available_actions)
        rewards_evil, rewards_good       = self._convert_dict_to_two_2darrays(rewards)

        # reshape rewards
        rewards_evil = rewards_evil.reshape(self.env.num_evil, 1)
        rewards_good = rewards_good.reshape(self.env.num_good, 1)

        # Evil Buffer Update
        self.buffer_evil.insert(
            share_obs=share_obs_evil,
            obs=obs_evil,
            actions=actions_evil,
            action_log_probs=action_log_probs_evil,
            value_preds=values_evil,
            rewards=rewards_evil,
            available_actions=av_actions_evil,
        )
        # Good Buffer Update
        self.buffer_good.insert(
            share_obs=share_obs_good,
            obs=obs_good,
            actions=actions_good,
            action_log_probs=action_log_probs_good,
            value_preds=values_good,
            rewards=rewards_good,
            available_actions=av_actions_good,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        # Evil
        self.trainer_evil.prep_rollout()        
        next_values_evil = self.trainer_evil.policy.get_values(
            self.buffer_evil.share_obs[-1]
        )
        next_values_evil = np.array(next_values_evil)
        self.buffer_evil.compute_returns(next_values_evil)

        # Good
        self.trainer_good.prep_rollout()
        next_values_good = self.trainer_good.policy.get_values(
            self.buffer_good.share_obs[-1]
        )
        next_values_good = np.array(next_values_good)
        self.buffer_good.compute_returns(next_values_good)

    def train(self) -> Tuple[dict, dict]:
        """Train policies with data in buffer. """
        self.trainer_evil.prep_training()        
        train_infos_evil = self.trainer_evil.train(self.buffer_evil)

        self.trainer_good.prep_training()
        train_infos_good = self.trainer_good.train(self.buffer_good)

        return (train_infos_evil, train_infos_good)

    @torch.no_grad()
    def eval(self, num_games=10, deterministic=False, verbose=False):
        """
        episode_length: (int) Maximum length of an episode
        """
        counter_wins_good = 0
        counter_wins_evil = 0
        for episode in range(num_games):
            if verbose:
                print(f'Episode #{episode+1}')
            obs, share_obs, available_actions, info = self.env.reset()
            obs_evil, obs_good = self._convert_dict_to_two_2darrays(obs)
            share_obs_evil, share_obs_good = self._convert_dict_to_two_2darrays(share_obs)
            available_actions_evil, available_actions_good = self._convert_dict_to_two_2darrays(available_actions)

            if verbose:
                print(f'Good players: {self.env.good_players}')
                print(f'Evil players: {self.env.evil_players}')

            done = False
            counter_step = 0
            while not done:
                if verbose:
                    print(f'Round: {self.env.round}, Phase: {self.env.phase}')
                    if self.env.phase == 1:
                        print(f'Team to vote for: {self.env.quest_team}')

                # Evil: Get action from the trained policy            
                values_evil, actions_evil, action_log_probs_evil, action_probs_evil = self.trainer_evil.policy.get_actions(
                    share_obs_evil,  # wrap in batch dimension
                    obs_evil,
                    available_actions_evil,
                    deterministic=deterministic
                )

                # Good: Get action from the trained policy            
                values_good, actions_good, action_log_probs_good, action_probs_good = self.trainer_good.policy.get_actions(
                    share_obs_good,  # wrap in batch dimension
                    obs_good,
                    available_actions_good,
                    deterministic=deterministic
                )

                actions_env = self._convert_to_act_dict(actions_evil, actions_good)

                # Step environment
                obs, share_obs, available_actions, rewards, done, info = self.env.step(actions_env)
                obs_evil, obs_good = self._convert_dict_to_two_2darrays(obs)
                share_obs_evil, share_obs_good = self._convert_dict_to_two_2darrays(share_obs)
                available_actions_evil, available_actions_good = self._convert_dict_to_two_2darrays(available_actions)               
                counter_step += 1

                # print('!========')
                # print(actions_env)
                # print('!========')

                if done:
                    if self.env.good_victory:
                        counter_wins_good += 1
                    else:
                        counter_wins_evil += 1

            #print(f"Episode {episode + 1}, Number of steps: {counter_step}")
            #print('=======  EPISODE END  ================')
            #print('')

        print(f"Good Win Rate: {counter_wins_good / num_games}")
        print(f"Evil Win Rate: {counter_wins_evil / num_games}")
    
    @torch.no_grad()
    def eval_good(self, num_games=100):
        counter_wins_good = 0
        number_of_turns = 0
        number_of_success = 0
        game_config = {
            0: self.trainer_good.policy,
            1: NaiveEvilPolicy()
        }
        env = ResistanceFinalEnv()
        for episode in range(num_games):
            stat = simulate_game(env, game_config)
            counter_wins_good += stat['good victory']
            number_of_turns  += stat['number of turns']
            if stat['good victory'] == 1:
                number_of_success += 3
            else:
                number_of_success += (stat['number of rounds'] - 3)
        print('Good Victories Rate', counter_wins_good / num_games)
        print('Av number of turns', number_of_turns/ num_games)
        print('Av number of succ Quests', number_of_success/ num_games)



    def save(self, version=''):
        """
        Save policy's actor and critic networks
        """
        # Evil
        policy_actor_evil = self.trainer_evil.policy.actor
        torch.save(policy_actor_evil.state_dict(), str(self.save_dir) + f"/actor_evil{version}.pt")
        policy_critic_evil = self.trainer_evil.policy.critic
        torch.save(policy_critic_evil.state_dict(), str(self.save_dir) + f"/critic_evil{version}.pt")

        # Good
        policy_actor_good = self.trainer_good.policy.actor
        torch.save(policy_actor_good.state_dict(), str(self.save_dir) + f"/actor_good{version}.pt")
        policy_critic_good = self.trainer_good.policy.critic
        torch.save(policy_critic_good.state_dict(), str(self.save_dir) + f"/critic_good{version}.pt")

    def restore(self, model_dir, version=''):
        """
        Restore policy's networks from a saved model.
        """
        policy_actor_state_dict_evil = torch.load(str(model_dir) + f'/actor_evil{version}.pt')
        self.policy_evil.actor.load_state_dict(policy_actor_state_dict_evil)
        policy_critic_state_dict_evil = torch.load(str(model_dir) + f'/critic_evil{version}.pt')
        self.policy_evil.critic.load_state_dict(policy_critic_state_dict_evil)

        policy_actor_state_dict_good = torch.load(str(model_dir) + f'/actor_good{version}.pt')
        self.policy_good.actor.load_state_dict(policy_actor_state_dict_good)
        policy_critic_state_dict_good = torch.load(str(model_dir) + f'/critic_good{version}.pt')
        self.policy_good.critic.load_state_dict(policy_critic_state_dict_good)
    
    def _convert_to_act_dict(self, actions_evil: torch.Tensor, actions_good: torch.Tensor) -> dict:
        flattened_data_evil = actions_evil.flatten()
        flattened_data_good = actions_good.flatten()
        action_dict_evil = {agent_id: value.item() for agent_id, value in zip(self.env.evil_players, flattened_data_evil)}
        action_dict_good = {agent_id: value.item() for agent_id, value in zip(self.env.good_players, flattened_data_good)}
        action_dict = {}
        for agent_id in self.env._agent_ids:
            if agent_id in action_dict_evil:
                action_dict[agent_id] = action_dict_evil[agent_id]
            elif agent_id in action_dict_good:
                action_dict[agent_id] = action_dict_good[agent_id]
        return action_dict
    
    def _convert_player_to_vector(self, player: str) -> np.ndarray:
        player_idx = int(player[-1]) - 1
        player_vector = [0] * 5
        player_vector[player_idx] = 1
        return np.array(player_vector, dtype=np.float32)
    
    def _get_available_actions_for_good(self, agent_id) -> np.ndarray:
        mask = [0] * 24
        agent_idx = int(agent_id[-1]) - 1  # 0 for player_1        

        if self.env.leader == agent_id and self.env.phase == 0:
            if self.env.round == 0 or self.env.round == 2:
                mask[1:11] = [1] * 10
            else:
                mask[11:21] = [1] * 10
        elif self.env.phase == 1:
            mask[21:23] = [1] * 2        
        elif self.env.phase == 2 and self.env.quest_team[agent_idx] == 1:
            mask[23] = 1
        else:
            mask[0] = 1        
        return np.array(mask, dtype=np.float32)
    
    def _convert_dict_to_two_2darrays(self, player_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
        # Divide a dictionary to numpy arrays for Good and Evil players
        evil_players_dict = {p:v for p, v in player_dict.items() if p in self.env.evil_players }
        good_players_dict = {p:v for p, v in player_dict.items() if p in self.env.good_players }

        values_evil = list(evil_players_dict.values())
        values_good = list(good_players_dict.values())
        return np.array(values_evil), np.array(values_good)




