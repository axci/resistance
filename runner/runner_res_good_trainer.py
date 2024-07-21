import time
import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils.buffer import SharedReplayBuffer
from alg.mappo import MAPPOTrainer, MAPPOPolicy


class RunnerMA_Good:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['env']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.learned_evil_policy = config['evil_policy']

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
        for episode in tqdm(range(episodes)):
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
                # We have actions_env for Good players, we need to add Evil Players from learned policy
                # ! But observations are different!
                # Let's invoke Evil observation from observation batch
                share_obs_evil = self.buffer.share_obs[step][0][0] # we need to add knowledge
                knowledge = self._get_knowledge()
                share_obs_evil = np.concatenate((share_obs_evil, knowledge)).reshape(1, 1, 420)

                # Let's add actions one by one
                for evil_player in self.env.evil_players:
                    obs_evil = self.buffer.obs[step][0][0][5:]  # except the the first 5 elemnts - player vector id
                    # now we need to add player vector id
                    player_vector_id = self._convert_player_to_vector(evil_player)

                    obs_evil = np.concatenate((player_vector_id, obs_evil, knowledge)).reshape(1, 1, 425)
                    available_actions_evil = self._get_available_actions_for_evil(evil_player)
                    available_actions_evil = available_actions_evil.reshape(1, 1, 25)
                    _, actions_evil, _, _ = self.learned_evil_policy.get_actions(
                        share_obs_evil,
                        obs_evil,
                        available_actions_evil
                    )
                    action_evil = actions_evil.item()  # convert to scalar

                    # add evil players' actions
                    actions_env[evil_player] = action_evil                

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
        actions_env: dict = self._convert_to_act_dict(actions)
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
    def eval(self, num_games=10, deterministic=False, verbose=False, verbose_render=False):
        """
        episode_length: (int) Maximum length of an episode

        """
        total_rewards = []
        counter_wins_good = 0
        counter_wins_evil = 0
        for episode in range(num_games):
            if verbose:
                print(f'Episode #{episode+1}')
            obs, share_obs, available_actions, info = self.env.reset()
            if verbose:
                print(f'Good players: {self.env.good_players}')
                print(f'Evil players: {self.env.evil_players}')

            done = False
            episode_reward = 0
            counter_step = 0
            while not done:
                if verbose:
                    print(f'Round: {self.env.round}, Phase: {self.env.phase}')
                    if self.env.phase == 1:
                        print(f'Team to vote for: {self.env.quest_team}')

                # Get action from the trained policy            
                values, actions, action_log_probs, action_probs = self.trainer.policy.get_actions(
                    share_obs,  # wrap in batch dimension
                    obs,
                    available_actions,
                    deterministic=deterministic
                )
                actions_env = self._convert_to_act_dict(actions)

                ####################################### Compute actions for Evil Players
                knowledge = self._get_knowledge()
                share_obs_evil = share_obs.copy()  # except the last 5 elements - knowledge
                share_obs_evil = np.concatenate((share_obs_evil, knowledge))

                # Let's add actions one by one
                for evil_player in self.env.evil_players:
                    obs_evil = obs.copy()[0][5:]  # except player vector id
                    player_vector_id = self._convert_player_to_vector(evil_player)
                    obs_evil = np.concatenate((player_vector_id, obs_evil, knowledge))
                    available_actions_evil = self._get_available_actions_for_evil(evil_player)
                    _, actions_evil, _, _ = self.learned_evil_policy.get_actions(
                        share_obs_evil,
                        obs_evil,
                        available_actions_evil
                    )
                    action_evil = actions_evil.item()  # convert to scalar

                    # add evil players' actions
                    actions_env[evil_player] = action_evil       
                ####################################### 

                # Step environment
                obs, share_obs, available_actions, rewards, done, info = self.env.step(actions_env)                
                episode_reward += np.sum(rewards)
                counter_step += 1

                if done:
                    if self.env.good_victory:
                        counter_wins_good += 1
                    else:
                        counter_wins_evil += 1
                    
                if verbose:
                    user_friendly_format = [[f"{num*100:.0f}%" for num in row] for row in action_probs.tolist()]
                    for player, action in actions_env.items():
                        if action != 0:
                            act_display = self.env.mapping_dict_action_to_team[action]
                            print(f'{player}: {act_display}')
                    print(f'Action: {actions_env}, action probability: {user_friendly_format}')
                    print('=======')
                if verbose_render:
                    self.env.render()

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}, Number of steps: {counter_step}")
            print('=======  EPISODE END  ================')
            print('')

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_games} episodes: {avg_reward}")
        print(f"Good Win Rate: {counter_wins_good / num_games}")
        print(f"Evil Win Rate: {counter_wins_evil / num_games}")

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
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor_good.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(model_dir) + '/critic_good.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def _convert_to_act_dict(self, actions: torch.Tensor) -> dict:
        flattened_data = actions.flatten()
        action_dict = {agent_id: value.item() for agent_id, value in zip(self.env.good_players, flattened_data)}
        return action_dict
    
    def _convert_player_to_vector(self, player: str) -> np.ndarray:
        player_idx = int(player[-1]) - 1
        player_vector = [0] * 5
        player_vector[player_idx] = 1
        return np.array(player_vector, dtype=np.float32)
    
    def _get_knowledge(self) -> np.ndarray:
        """
        Returns knowledge vector according to player's indicies:
            1: Evil player
            0: Good player
        Example return:
            [1, 0, 0, 1, 0] if player 1 and player 4 are Evil
        """
        knowledge = [0] * 5
        for evil_player in self.env.evil_players:
            evil_player_idx = int(evil_player[-1]) - 1
            knowledge[evil_player_idx] = 1
        return np.array(knowledge, dtype=np.float32)

    def _get_available_actions_for_evil(self, agent_id) -> np.ndarray:
        mask = [0] * 25
        agent_idx = int(agent_id[-1]) - 1  # 0 for player_1        

        if self.env.leader == agent_id and self.env.phase == 0:
            if self.env.round == 0 or self.env.round == 2:
                mask[1:11] = [1] * 10
            else:
                mask[11:21] = [1] * 10
        elif self.env.phase == 1:
            mask[21:23] = [1] * 2        
        elif self.env.phase == 2 and self.env.quest_team[agent_idx] == 1:
            mask[23:25] = [1] * 2
        else:
            mask[0] = 1
        return np.array(mask, dtype=np.float32)




