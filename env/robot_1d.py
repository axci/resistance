import numpy as np
from gym.spaces import MultiBinary, Discrete


class CorridorEnv:
    """
    Arguments:
        observation_space: (gym.MultiBinary):
            2: is wall on the left? is wall on the right?
            3: coin on the left? coin on the same cell? coin on the right?
            1: is coin found?
            2: is left wall found? is right wall found
            = 8
        action_space(gym.Discrete):
            0: stay
            1: go left
            2: go right
            3: pick up Coin
    """
    def __init__(self, num_cells, num_agents, robot_position=None, set_manually=False):
        self.num_agents = num_agents
        self.num_cells = num_cells
        self.robot_position = robot_position
        self.set_manually = set_manually
        self.coin_position = None
        self.coin_reward = 100
        self.walk_reward = -1
        self.stay_reward = -1
        self.observation_space = MultiBinary(8)
        self.action_space = Discrete(4)

    def reset(self):
        if not self.set_manually:
            self.robot_position = np.random.choice(range(self.num_cells))    #self.num_cells // 2
        self.coin_position = np.random.choice(range(self.num_cells))    #self.robot_position // 2
        self.is_coin_collected = False        
        self.is_right_wall_found = False
        self.is_left_wall_found = False
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        reward = 0
        done = False
        if action == 0:
            reward = self.stay_reward
        elif action == 1:  # go left
            self.robot_position -= 1
            reward = self.walk_reward
        elif action == 2:  # go right
            self.robot_position += 1
            reward = self.walk_reward
        elif action == 3:
            self.is_coin_collected = True
            reward = self.coin_reward
            #print('======= Coin is collected.')
        if self.is_coin_collected:
            done = True

        obs = self._get_obs()
        return obs, reward, done

    def _get_obs(self):
        if self.num_cells == 1:
            wall = [1, 1]
            self.is_left_wall_found = True
            self.is_right_wall_found = True
        elif self.robot_position == 0:
            wall = [1, 0]
            self.is_left_wall_found = True
        elif self.robot_position == self.num_cells - 1:
            wall = [0, 1]
            self.is_right_wall_found = True
        else:
            wall = [0, 0]
        
        # COIN
        if not self.is_coin_collected:
            if self.robot_position - self.coin_position == 0:
                coin = [0, 1, 0]
            elif self.robot_position - self.coin_position == 1:
                coin = [1, 0, 0]
            elif self.robot_position - self.coin_position == -1:
                coin = [0, 0, 1]
            else:
                coin = [0, 0, 0]
        else:
            coin = [0, 0, 0]
        
        is_coin_collected = [1 if self.is_coin_collected else 0 ]
        left_wall = [1 if self.is_left_wall_found else 0 ]
        right_wall = [1 if self.is_right_wall_found else 0 ]
       
        obs = wall + coin + is_coin_collected + left_wall + right_wall
        return np.array(obs)
    
    def get_available_actions(self, obs) -> np.ndarray:
        """
        Example return:
            [1, 0, 1, 0] # you can not go to the left, you can not pick up a coin

        """
        mask = [1, 1, 1, 1]  # [0]: stay, [1]: go left, [2]: go right, [3]: pick up a coin
        if obs[0] == 1:  # there is a wall on the left
            mask[1] = 0 
        if obs[1] == 1:  # there is a wall on the right
            mask[2] = 0
        if obs[3] != 1:  # there is a NO coin on the same cell as you
            mask[3] = 0     
        return np.array(mask)


class CorridorMultiAgentEnv:
    """
    Arguments:
        observation_space: (gym.MultiBinary):
            2: is wall on the left? is wall on the right?
            3: coin on the left? coin on the same cell? coin on the right?
            1: is coin found?
            2: is left wall found? is right wall found
            = 8
        action_space(gym.Discrete):
            0: skip
            1: stay
            2: go left
            3: go right
            4: pick up Coin
    """
    def __init__(self, num_cells, num_agents):
        self.num_agents = num_agents
        self._agent_ids = [f'robot_{i+1}' for i in range(self.num_agents)]


        self.num_cells = num_cells
        self.robot_position = { agent_id: None for agent_id in self._agent_ids }
        self.coin_position = None
        self.coin_reward = 100
        self.walk_reward = -1
        self.stay_reward = -1
        self.observation_space = MultiBinary(13)
        self.share_observation_space = MultiBinary(12) 

        self.action_space = Discrete(5)

    def reset(self):
        positions_list_so_far = []
        for agent_id, position in self.robot_position.items():
            self.robot_position[agent_id] = np.random.choice( [i for i in range(self.num_cells) if i not in positions_list_so_far] )
            positions_list_so_far.append(self.robot_position[agent_id])

        self.coin_position = np.random.choice(range(self.num_cells))   
        self.is_coin_collected = False        
        self.is_right_wall_found     =  {agent_id: False for agent_id in self._agent_ids}
        self.is_left_wall_found      =  {agent_id: False for agent_id in self._agent_ids}
        self.is_found_agent_on_left  =  {agent_id: False for agent_id in self._agent_ids}
        self.is_found_agent_on_right =  {agent_id: False for agent_id in self._agent_ids}

        self.if_meeting = False  # if robots found each other

        self.active_robot = np.random.choice(self._agent_ids)
        obs = {
            agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids
            }
        
        available_actions = {
            agent_id: self.get_available_actions(obs[agent_id]) for agent_id in self._agent_ids
        }
        
        share_obs = self._get_share_obs(obs)  
        info = self._get_info(obs, available_actions)                    
        
        # convert dict to numpy arrays
        obs_array = self._convert_dict_2d_array(obs)
        available_actions_array = self._convert_dict_2d_array(available_actions)

        return obs_array, share_obs, available_actions_array, info #, share_obs
    
    def step(self, action_dict):
        rewards = {
            agent_id: 0 for agent_id in self._agent_ids
        }
        done = False
        # Perform actions:
        for agent_id, action in action_dict.items():
            if action == 1:  # stay
                rewards[agent_id] = self.stay_reward
            elif action == 2:  # go left
                self.robot_position[agent_id] -= 1
                rewards[agent_id] = self.walk_reward
            elif action == 3:  # go right
                self.robot_position[agent_id] += 1
                rewards[agent_id] = self.walk_reward
            elif action == 4:
                self.is_coin_collected = True
                #rewards[agent_id] = self.coin_reward
                #print('======= !!!!!!!! Coin is collected.')
        
        if self.is_coin_collected:
            rewards = {
                agent_id: self.coin_reward / self.num_agents for agent_id in self._agent_ids
            }
            done = True
        
        self._change_active_player()
        
        # Get obs:
        obs = {
            agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids
            }
        
        available_actions = {
            agent_id: self.get_available_actions(obs[agent_id]) for agent_id in self._agent_ids
        }

        share_obs = self._get_share_obs(obs)  
        info = self._get_info(obs, available_actions)

        # convert to numpy arrays
        obs_array = self._convert_dict_2d_array(obs)
        available_actions_array = self._convert_dict_2d_array(available_actions)

        # convert from 1d to 2d: array([1,0]) - > array([[1], [0]])
        rewards_array = np.reshape(self._convert_dict_2d_array(rewards), (2, 1)  )

        return obs_array, share_obs, available_actions_array, rewards_array, done, info

    def _change_active_player(self):
        act_robot_index = int(self.active_robot[-1]) - 1
        next_active_robot_index = (act_robot_index + 1) % self.num_agents
        self.active_robot = f'robot_{next_active_robot_index+1}'

    def _get_obs(self, agent_id: str) -> list:
        #WALL
        if self.num_cells == 1:
            wall = [1, 1]
            self.is_left_wall_found[agent_id] = True
            self.is_right_wall_found[agent_id] = True
        elif self.robot_position[agent_id] == 0:
            wall = [1, 0]
            self.is_left_wall_found[agent_id] = True
        elif self.robot_position[agent_id] == self.num_cells - 1:
            wall = [0, 1]
            self.is_right_wall_found[agent_id] = True
        else:
            wall = [0, 0]

        # COIN
        if not self.is_coin_collected:
            if self.robot_position[agent_id] - self.coin_position == 0:
                coin = [0, 1, 0]
            elif self.robot_position[agent_id] - self.coin_position == 1:
                coin = [1, 0, 0]
            elif self.robot_position[agent_id] - self.coin_position == -1:
                coin = [0, 0, 1]
            else:
                coin = [0, 0, 0]
        else:
            coin = [0, 0, 0]

        # ANOTHER AGENTS
        other_robots = [0, 0]
        robot_positions = [ position for agent_id, position in self.robot_position.items() ]
        for robot_position in robot_positions:
            if self.robot_position[agent_id] - robot_position == 1:
                other_robots[0] = 1 # there is a robot on the left
                self.is_found_agent_on_left[agent_id] = True
            if self.robot_position[agent_id] - robot_position == -1:
                other_robots[1] = 1 # there is a robot on the right
                self.is_found_agent_on_right[agent_id] = True
        
        is_coin_collected = [1 if self.is_coin_collected else 0 ]
        left_wall = [1 if self.is_left_wall_found[agent_id] else 0 ]
        right_wall = [1 if self.is_right_wall_found[agent_id] else 0 ]
        
        left_agent = [1 if self.is_found_agent_on_left[agent_id] else 0 ]
        right_agent = [1 if self.is_found_agent_on_right[agent_id] else 0 ]

        # is it your turn?
        your_turn = [0]
        if self.active_robot == agent_id:
            your_turn = [1]

        obs = wall + coin + other_robots + is_coin_collected + left_wall + right_wall + left_agent + right_agent + your_turn
        return obs
    
    def _get_share_obs(self, obs: dict) -> np.ndarray:
        """
        obs
        """
        assert self.num_agents == 2,' Only implemented for 2 players'
        share_obs = [0] * 4
        whos_turn = []
        coin_seeing = []
        for agent_id, observation in obs.items():
            if observation[8] == 1:
                share_obs[0] = 1  # left wall is found
            if observation[9] == 1:
                share_obs[1] = 1  # right wall is found
            if observation[10] == 1 or observation[11] == 1:
                share_obs[2] = 1  # agents already met
            if observation[7] == 1:
                share_obs[3]  # coin is collected
            whos_turn.append(observation[12])
            coin_seeing.append(observation[2:5])
        cs = coin_seeing[0] + coin_seeing[1]  # from [[0, 0, 1], [0, 0, 0]] to [0,0,1,0,0,0]
        
        return np.array(share_obs + whos_turn + cs)        

    def _get_info(self, obs, available_actions):
        info = {
            agent_id: {
                'walls': obs[agent_id][:2],
                'coin': obs[agent_id][2:5],
                'another_robot': obs[agent_id][5:7],
                'is coin collected': obs[agent_id][7],
                'found walls': obs[agent_id][8:10],
                'found agents': obs[agent_id][10:12],
                'your turn': obs[agent_id][12],
                'avail.actions': available_actions[agent_id],
                "share_obs": self._get_share_obs(obs),
            } for agent_id in self._agent_ids
            }
        return info

    def get_available_actions(self, obs) -> np.ndarray:
        """
        mask indicies:
            0: skip (it's not your turn)
            1: stay
            2: go left
            3: go right
            4: pick up the Coin
        """
        mask = [1, 1, 1, 1, 1]  
        if obs[-1] == 1:  # if it's your turn
            mask[0] = 0
            if obs[0] == 1 or obs[5]:  # there is a wall OR robot on the left
                mask[2] = 0 
            if obs[1] == 1 or obs[6]:  # there is a wall OR robot on the right
                mask[3] = 0
            if obs[3] != 1:  # there is a NO coin on the same cell as you
                mask[4] = 0
        else:
            mask = [1, 0, 0, 0, 0]
            
        return np.array(mask)
    
    def _convert_dict_2d_array(self, di):
        # Extract the lists (values) from the dictionary
        values = list(di.values())

        return np.array(values)
    
    def render(self):
        corridor = ['.'] * self.num_cells
        if self.coin_position is not None and not self.is_coin_collected:
            corridor[self.coin_position] = 'C'
        
        for agent_id, position in self.robot_position.items():
            if corridor[position] == '.':
                corridor[position] = agent_id[-1]
            else:
                corridor[position] += agent_id[-1]

    
        corridor_str = '|'.join(corridor)
        print(corridor_str)


class CorridorEnv_v2:
    """
    Arguments:
        observation_space: (gym.MultiBinary):
            2: is wall on the left? is wall on the right?
            3: coin on the left? coin on the same cell? coin on the right?
            1: is coin found?
            2: is left wall found? is right wall found
            = 8
        action_space(gym.Discrete):
            0: stay
            1: go left
            2: go right
            3: pick up Coin
    """
    def __init__(self, num_cells, num_agents, robot_position=None, set_manually=False):
        self.num_agents = num_agents
        self.num_cells = num_cells
        self.robot_position = robot_position
        self.set_manually = set_manually
        self.coin_position = None
        self.gold_position = None
        self.coin_reward = 100
        self.gold_reward = 200
        self.walk_reward = -1
        self.stay_reward = -1
        self.observation_space = MultiBinary(12)
        self.action_space = Discrete(5)

    def reset(self):
        if not self.set_manually:
            self.robot_position = np.random.choice(range(self.num_cells))    #self.num_cells // 2
        self.coin_position = np.random.choice(range(self.num_cells))    #self.robot_position // 2
        self.gold_position = np.random.choice(range(self.num_cells)) 
        self.is_coin_collected = False
        self.is_gold_collected = False
        self.is_right_wall_found = False
        self.is_left_wall_found = False
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        reward = 0
        done = False
        if action == 0:
            reward = self.stay_reward
        elif action == 1:  # go left
            self.robot_position -= 1
            reward = self.walk_reward
        elif action == 2:  # go right
            self.robot_position += 1
            reward = self.walk_reward
        elif action == 3:
            self.is_coin_collected = True
            reward = self.coin_reward
            #print('======= Coin is collected.')
        elif action == 4:
            self.is_gold_collected = True
            reward = self.gold_reward
            #print('======= Gold is collected.')
        
        # Termination condition
        if self.is_coin_collected and self.is_gold_collected:
            done = True

        obs = self._get_obs()
        return obs, reward, done

    def _get_obs(self):
        if self.num_cells == 1:
            wall = [1, 1]
            self.is_left_wall_found = True
            self.is_right_wall_found = True
        elif self.robot_position == 0:
            wall = [1, 0]
            self.is_left_wall_found = True
        elif self.robot_position == self.num_cells - 1:
            wall = [0, 1]
            self.is_right_wall_found = True
        else:
            wall = [0, 0]
        
        # COIN
        if not self.is_coin_collected:
            if self.robot_position - self.coin_position == 0:
                coin = [0, 1, 0]
            elif self.robot_position - self.coin_position == 1:
                coin = [1, 0, 0]
            elif self.robot_position - self.coin_position == -1:
                coin = [0, 0, 1]
            else:
                coin = [0, 0, 0]
        else:
            coin = [0, 0, 0]

        # GOLD
        if not self.is_gold_collected:
            if self.robot_position - self.gold_position == 0:
                gold = [0, 1, 0]
            elif self.robot_position - self.gold_position == 1:
                gold = [1, 0, 0]
            elif self.robot_position - self.gold_position == -1:
                gold = [0, 0, 1]
            else:
                gold = [0, 0, 0]
        else:
            gold = [0, 0, 0]
        
        is_coin_collected = [1 if self.is_coin_collected else 0 ]
        is_gold_collected = [1 if self.is_gold_collected else 0 ]
        left_wall = [1 if self.is_left_wall_found else 0 ]
        right_wall = [1 if self.is_right_wall_found else 0 ]
       
        obs = wall + coin + gold + is_coin_collected + is_gold_collected + left_wall + right_wall
        return np.array(obs)
    
    def get_available_actions(self, obs) -> np.ndarray:
        """
        obs = [0,0,   0,0,0,   0,0,0,   0,0,   0,0]
        """
        mask = [1, 1, 1, 1, 1]  # stay, go left, go right, pick up a coin, pick up a gold
        if obs[0] == 1:  # there is a wall on the left
            mask[1] = 0 
        if obs[1] == 1:  # there is a wall on the right
            mask[2] = 0
        if obs[3] != 1:  # there is a NO coin on the same cell as you
            mask[3] = 0
        if obs[6] != 1:  # there is a NO gold on the same cell as you
            mask[4] = 0 

        return np.array(mask)


class CorridorFullyVisibleEnv:
    """
    Arguments:
        observation_space: (gym.MultiBinary):
            2: is wall on the left? is wall on the right?
            3: coin on the left? coin on the same cell? coin on the right?
            1: is coin found?
            2: is left wall found? is right wall found
            = 8
        action_space(gym.Discrete):
            0: stay
            1: go left
            2: go right
            3: pick up Coin
    """
    def __init__(self, num_cells, num_agents):
        self.num_agents = num_agents
        self.num_cells = num_cells
        self.robot_position = None
        self.coin_position = None
        self.coin_reward = 100
        self.walk_reward = -1
        self.stay_reward = -1
        self.observation_space = MultiBinary(num_cells * 2)
        self.action_space = Discrete(4)

    def reset(self):
        self.robot_position = np.random.choice(range(self.num_cells))    #self.num_cells // 2
        self.coin_position = np.random.choice(range(self.num_cells))    #self.robot_position // 2
        self.is_coin_collected = False        
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        reward = 0
        done = False
        if action == 0:
            reward = self.stay_reward
        elif action == 1:  # go left
            self.robot_position -= 1
            reward = self.walk_reward
        elif action == 2:  # go right
            self.robot_position += 1
            reward = self.walk_reward
        elif action == 3:
            self.is_coin_collected = True
            reward = self.coin_reward
            #print('======= Coin is collected.')
        if self.is_coin_collected:
            done = True

        obs = self._get_obs()
        return obs, reward, done

    def _get_obs(self) -> np.ndarray:
        robot = [0] * self.num_cells
        robot[self.robot_position] = 1
        coin = [0] * self.num_cells
        coin[self.coin_position] = 1
        return np.array(robot + coin)
    
    def get_available_actions(self, obs) -> np.ndarray:
        mask = [1, 1, 1, 1]  # stay, go left, go right, pick up a coin
        if obs[0] == 1:
            mask[1] = 0
        if obs[self.num_cells - 1] == 1:
            mask[2] = 0
        robot_coin = np.where(obs==1)[0]
        if robot_coin[1] - robot_coin[0] != self.num_cells:
            mask[3] = 0
        return np.array(mask)
    
class RandomRobotEnv:
    def __init__(self, env):
        self.env = env

    def compute_single_action(self, obs, verbose=False):
        mask = self.env.get_available_actions(obs)
        available_actions = np.where(np.array(mask)==1)[0]
        action = np.random.choice(available_actions)
        if verbose:
            print('My observation:', obs)
            print("Available actions:", available_actions)
            print("My action:", action)
        return action




class RandomRobot:
    def __init__(self):
        pass

    def compute_single_action(self, obs, verbose=False):
        # obs [1, 0,   0, 0, 0,   0,   0, 0]
        mask = self._check_action_availability(obs)  # [1, 1, 0]
        available_actions = np.where(np.array(mask)==1)[0]
        
        action = np.random.choice(available_actions)
        if verbose:
            print('My observation:', obs)
            print("Available actions:", available_actions)
            print("My action:", action)
        return action
    
    def _check_action_availability(self, obs):
        mask = [1, 1, 1, 1]  # stay, go left, go right, pick up a coin, pick up a gold
        if obs[0] == 1:  # there is a wall on the left
            mask[1] = 0 
        if obs[1] == 1:  # there is a wall on the right
            mask[2] = 0
        if obs[3] != 1:  # there is a NO coin on the same cell as you
            mask[3] = 0     
        return mask