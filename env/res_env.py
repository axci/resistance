from gymnasium.spaces import Discrete, MultiBinary
import numpy as np
from itertools import chain
import random


class ResistanceEnv:
    def __init__(self):
        super(ResistanceEnv).__init__()
        """
        PHASES:
            (0) Start
            (1) Team Building
            (2) Team Voting
            (3) Quest Voting     

        observation_space:

        """
        self.mapping_dict_action_to_team = {
            1 : [1, 1, 0, 0, 0],
            2 : [1, 0, 1, 0, 0],
            3 : [1, 0, 0, 1, 0],
            4 : [1, 0, 0, 0, 1],
            5 : [0, 1, 1, 0, 0],
            6 : [0, 1, 0, 1, 0],
            7 : [0, 1, 0, 0, 1],
            8 : [0, 0, 1, 1, 0],
            9 : [0, 0, 1, 0, 1],
           10 : [0, 0, 0, 1, 1],
           11 : [1, 1, 1, 0, 0],
           12 : [1, 1, 0, 1, 0],
           13 : [1, 1, 0, 0, 1],
           14 : [1, 0, 1, 1, 0],
           15 : [1, 0, 1, 0, 1],
           16 : [1, 0, 0, 1, 1],
           17 : [0, 1, 1, 1, 0],
           18 : [0, 1, 1, 0, 1],
           19 : [0, 1, 0, 1, 1],
           20 : [0, 0, 1, 1, 1],
           21 : 1,  # Vote for the proposed Team
           22 : 0,  # Vote against a proposed Team
           23 : 0,  # Vote for Success in Quest
        }

        self.num_players = 5
        self.num_good = 3
        self.num_evil = self.num_players - self.num_good
        self.num_players_for_quest = [2, 3, 2, 3, 3]
        self.num_fails_for_quest   = [1, 1, 1, 1, 1]
        self.reward_win_bonus = 10
        self.reward_completed_quest_bonus = 1

        self._agent_ids = [f"player_{i}" for i in range(1, self.num_players + 1)]
        #self.roles: dict = self.assign_roles()
        
        self.round = 0
        self.phase = 0

        self.observation_space = MultiBinary(420)
        self.share_observation_space = MultiBinary(415)
        self.action_space = Discrete(24)

    def reset(self):
        self.good_players = self.assign_roles()
        self.evil_players = [agent_id for agent_id in self._agent_ids if agent_id not in self.good_players]

        self.leader = np.random.choice(self._agent_ids)
        self.round = 0
        self.phase = 0
        self.voting_attempt = 1 
        self.quest_team = [0] * self.num_players #np.zeros(self.num_players)
        self.team_votes = [0] * self.num_players #np.zeros(self.num_players)
        self.voting_attempt = 1
        self.cum_quest_fails = 0
        self.history = self._get_start_history()
        self.good_victory = False
        self.done = False
        self.turn_count = 0

        obs = { agent_id: self._get_obs(agent_id) for agent_id in self.good_players }
        available_actions = { agent_id: self._get_available_actions(agent_id) for agent_id in self.good_players }
        share_obs = self._get_share_obs()  
        #share_obs = self._get_share_obs(obs)  
        info = self._get_info(obs) 

        # Convert to 2-d numpy array:
        obs_array = self._convert_dict_2d_array(obs)
        available_actions_array = self._convert_dict_2d_array(available_actions)

        return obs_array, share_obs, available_actions_array, info #, share_obs, available_actions, info
    
    def step(self, action_dict, verbose=False):
        rewards = {
            agent_id: 0 for agent_id in self.good_players
        }

        if self.phase == 0:
            if sum(action_dict.values()):  # if 1 of the agents is active
                for agent_id, action in action_dict.items():
                    assert agent_id in self.good_players, f"{agent_id} is not a Good player!"
                    if action != 0:
                        team = self._action_to_array(action)
                        if verbose:
                            print(f'Good player chose action {action} - team {team}')

                        self.phase_0(team, verbose=verbose)
            else:  # predefinded action for Evil Team
                team = self._get_predifined_evil_action_phase0()
                self.phase_0(team, verbose=verbose)
        
        elif self.phase == 1:
            vote_action = [0] * 5
            for agent_id, action in action_dict.items():
                agent_idx = int(agent_id[-1]) - 1                
                vote_action[agent_idx] = self.mapping_dict_action_to_team[action]
            # predefined actions for Evil Team
            evil_action_dict = self._get_predefined_evil_action_phase1()
            for agent_id, action in evil_action_dict.items():
                agent_idx = int(agent_id[-1]) - 1
                vote_action[agent_idx] = action
            
            if verbose:
                print('Team Votes:', vote_action)
            self.phase_1(vote_action, verbose=verbose)
        
        elif self.phase == 2:
            vote_action = 0
            for agent_id, action in action_dict.items():
                if action != 0:  # if it's NOT 'skip'
                    vote_action += self.mapping_dict_action_to_team[action]
            # predefined actions for Evil Team
            evil_action_dict = self._get_predefined_evil_action_phase1()
            for agent_id, action in evil_action_dict.items():
                vote_action += action
            self.phase_2(vote_action, verbose=verbose)
            if vote_action == 0:
                rewards = { agent_id: self.reward_completed_quest_bonus for agent_id in self.good_players }
            else:
                rewards = { agent_id: -self.reward_completed_quest_bonus for agent_id in self.good_players }

        self.turn_count += 1

        if self.done:
            if self.good_victory:
                rewards = { agent_id: self.reward_win_bonus for agent_id in self.good_players }
            else:
                rewards = { agent_id: -self.reward_win_bonus for agent_id in self.good_players }
        
        # Get obs and available actions
        obs = { agent_id: self._get_obs(agent_id) for agent_id in self.good_players }
        available_actions = { agent_id: self._get_available_actions(agent_id) for agent_id in self.good_players }
        share_obs = self._get_share_obs()

        # convert to numpy arrays
        obs_array = self._convert_dict_2d_array(obs)
        available_actions_array = self._convert_dict_2d_array(available_actions)
        # convert from 1d to 2d: array([1,0]) - > array([[1], [0]])
        rewards_array = np.reshape(self._convert_dict_2d_array(rewards), (3, 1)  )
        info = None
        return obs_array, share_obs, available_actions_array, rewards_array, self.done, info

    def _get_obs(self, agent_id: str) -> list:
        """
        Encoding:
            id: [1, 0, 0, 0, 0]  for player_1    
        """
        idx = self._encoding_player(agent_id)

        # Flatten the nested lists into a one-dimensional list
        history_flattened = list(chain.from_iterable(
            chain.from_iterable(round_data.values()) for round_data in self.history.values()
        ))
        return idx + history_flattened

    def _get_share_obs(self) -> np.ndarray:

        # Flatten the nested lists into a one-dimensional list
        history_flattened = list(chain.from_iterable(
            chain.from_iterable(round_data.values()) for round_data in self.history.values()
        ))
        return np.array(history_flattened)


    def _get_available_actions(self, agent_id) -> np.ndarray:
        mask = [0] * self.action_space.n
        agent_idx = int(agent_id[-1]) - 1  # 0 for player_1        

        if self.leader == agent_id and self.phase == 0:
            if self.round == 0 or self.round == 2:
                mask[1:11] = [1] * 10
            else:
                mask[11:21] = [1] * 10
        elif self.phase == 1:
            mask[21:23] = [1] * 2        
        elif self.phase == 2 and self.quest_team[agent_idx] == 1:
            mask[23] = 1
        else:
            mask[0] = 1
        
        return np.array(mask)




    def _get_info(self, obs, available_actions=None):
        """
        Player obs:
        round   lea1 team1 tv1    lea2  team2 tv2    lea3 team3 tv3   lea4  team4  tv4  lea5  team5 tv5   qv
        10000  01000 11000 11000  00100 01010 11001 00000 00000 00000 00000 00000 00000 00000 00000 00000 001
        01000  01000 11000 10000  00100 01010 11101 00000 00000 00000 00000 00000 00000 00000 00000 00000 000  
        00000  00000 00000 00000  00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 000
        00000  00000 00000 00000  00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 000
        00000  00000 00000 00000  00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 000

        """
        info = {
            agent_id: {
                'idx': obs[agent_id][:5],
                'round 0': {
                    'round_id': obs[agent_id][5:10],
                    'leader1': obs[agent_id][10:15],
                    'team1': obs[agent_id][15:20],
                    't_vote1': obs[agent_id][20:25],
                    'leader2': obs[agent_id][25:30],
                    'team2': obs[agent_id][30:35],
                    't_vote2': obs[agent_id][35:40],
                    'leader3': obs[agent_id][40:45],
                    'team3': obs[agent_id][45:50],
                    't_vote3': obs[agent_id][50:55],
                    'leader4': obs[agent_id][55:60],
                    'team4': obs[agent_id][60:65],
                    't_vote4': obs[agent_id][65:70],
                    'leader5': obs[agent_id][70:75],
                    'team5': obs[agent_id][75:80],
                    't_vote5': obs[agent_id][80:85],
                    'q_vote': obs[agent_id][85:88],
                    },
                'round 1': {
                    'round_id': obs[agent_id][88:93],
                    'leader1': obs[agent_id][93:98],
                    'team1': obs[agent_id][98:103],
                    't_vote1': obs[agent_id][103:108],
                    'leader2': obs[agent_id][108:113],
                    'team2': obs[agent_id][113:118],
                    't_vote2': obs[agent_id][118:123],
                    'leader3': obs[agent_id][123:128],
                    'team3': obs[agent_id][128:138],
                    't_vote3': obs[agent_id][133:138],
                    'leader4': obs[agent_id][138:143],
                    'team4': obs[agent_id][143:148],
                    't_vote4': obs[agent_id][148:153],
                    'leader5': obs[agent_id][153:158],
                    'team5': obs[agent_id][158:163],
                    't_vote5': obs[agent_id][163:168],
                    'q_vote': obs[agent_id][168:171],
                    },
                'round 2': {
                    'round_id': obs[agent_id][171:176],
                    'leader1': obs[agent_id][176:181],
                    'team1': obs[agent_id][181:186],
                    't_vote1': obs[agent_id][186:191],
                    'leader2': obs[agent_id][191:196],
                    'team2': obs[agent_id][196:201],
                    't_vote2': obs[agent_id][201:206],
                    'leader3': obs[agent_id][206:211],
                    'team3': obs[agent_id][211:216],
                    't_vote3': obs[agent_id][216:221],
                    'leader4': obs[agent_id][221:226],
                    'team4': obs[agent_id][226:231],
                    't_vote4': obs[agent_id][231:236],
                    'leader5': obs[agent_id][236:241],
                    'team5': obs[agent_id][241:246],
                    't_vote5': obs[agent_id][246:251],
                    'q_vote': obs[agent_id][251:254],
                },
                'round 3': {
                    'round_id': obs[agent_id][254:259],
                    'leader1': obs[agent_id][259:264],
                    'team1': obs[agent_id][264:269],
                    't_vote1': obs[agent_id][269:274],
                    'leader2': obs[agent_id][274:279],
                    'team2': obs[agent_id][279:284],
                    't_vote2': obs[agent_id][284:289],
                    'leader3': obs[agent_id][289:294],
                    'team3': obs[agent_id][294:299],
                    't_vote3': obs[agent_id][299:304],
                    'leader4': obs[agent_id][304:309],
                    'team4': obs[agent_id][309:314],
                    't_vote4': obs[agent_id][314:319],
                    'leader5': obs[agent_id][319:324],
                    'team5': obs[agent_id][324:329],
                    't_vote5': obs[agent_id][329:334],
                    'q_vote': obs[agent_id][334:337],
                },
                'round 4': {
                    'round_id': obs[agent_id][337:342],
                    'leader1': obs[agent_id][342:347],
                    'team1': obs[agent_id][347:352],
                    't_vote1': obs[agent_id][352:357],
                    'leader2': obs[agent_id][357:362],
                    'team2': obs[agent_id][362:367],
                    't_vote2': obs[agent_id][367:372],
                    'leader3': obs[agent_id][372:377],
                    'team3': obs[agent_id][377:382],
                    't_vote3': obs[agent_id][382:387],
                    'leader4': obs[agent_id][387:392],
                    'team4': obs[agent_id][392:397],
                    't_vote4': obs[agent_id][397:402],
                    'leader5': obs[agent_id][402:407],
                    'team5': obs[agent_id][407:412],
                    't_vote5': obs[agent_id][412:417],
                    'q_vote': obs[agent_id][417:420],
                    },
            } for agent_id in self.good_players
            }
        return info


    def phase_0(self, team: list, verbose=False):
        """ 
        Decision maker: 1 player (Leader).
        Leader chooses the Team. The size of the Team depends on the current Round.
        """
        
        assert sum(team) == self.num_players_for_quest[self.round], f"The team size for the Round #{self.round+1} should be {self.num_players_for_quest[self.round]}, not {sum(team)}."
        assert self.phase == 0, f"The game must be in the Phase 0 (Team Building), but the game now is in the Phase {self.phase}."
        
        self.quest_team = team
        
        if verbose:
            print(f"ROUND #{self.round}. Phase: (0, Team Building):")
            print(f'Leader: {self.leader}. Chosen Team: {self.quest_team})')
            print("=========================================")

        # CHANGE HISTORY
        # Update Leader
        self.history[f"round {self.round}"][f"attempt {self.voting_attempt}"][:5] = self._encoding_player(self.leader)
        # Update Team
        self.history[f"round {self.round}"][f"attempt {self.voting_attempt}"][5:10] = team        
        
        # Proceed to the next phase
        self.phase += 1  # move to the next phase - Voting

        # Change Leader:
        self._change_leader()

    def phase_1(self, votes: list, verbose=False):
        assert len(votes) == self.num_players, f"Number of votes ({len(votes)}) does not match the number of players ({self.num_players})."
        assert self.phase == 1, f"The game is in the {self.phase} phase. It should be in the Voting Phase."
        
        self.team_votes = votes
        
        # strict majority?        
        if sum(votes) > self.num_players / 2:
            self.phase += 1  # move to the next phase
            
            # Update history
            self.history[f"round {self.round}"][f"attempt {self.voting_attempt}"][10:15] = votes
        
            self.voting_attempt = 1  # reset voting attempts
            
            if verbose:
                print(f"ROUND #{self.round}. Phase: (1, Team Voting):")
                print(f'Success. ({sum(votes)} / {self.num_players}) voted for the Team.')
                print(f'The next Phase: {self.phase}')
                print("=========================================")
        else:  # Unsuccessful Team Vote
            self.phase = 0
            # Update history      
            self.history[f"round {self.round}"][f"attempt {self.voting_attempt}"][10:15] = votes
            self.voting_attempt += 1

            if verbose:
                print(f"ROUND #{self.round}. Phase: (1, Team Voting):")
                print(f'Fail. {sum(votes)} / {self.num_players} players voted for the Team (should be majority).')
                print(f"The total number of failed elections in this Round: {self.voting_attempt - 1}.")
                print(f'The next Phase: {self.phase}.')
                print("=========================================")
            
            # Evil wins the game if 5 teams are rejected in a single round 
            if self.voting_attempt == 6:
                self.done = True  # XXXXXXXXXXX END OF THE GAME XXXXXXXXXXXX
                self.good_victory = False
                self.phase = 1

                # Update history
                #self.history[f"round_{self.round}_team_votes"][(self.voting_attempt - 1) * 5: (self.voting_attempt - 1) * 5 + 5] = votes
                if verbose:
                    print(f'== The end ==. 5 consecutive failed Votes. Evil wins 😈 ')

    def phase_2(self, fail_votes: int, verbose=False):
        """

        """
        assert self.phase == 2, f"The game must be in the phase 2 (Quest Voting), but the game now is in the phase {self.get_phase()}."
        # Update history
        encoded_quest_vote = self._encoding_quest_vote(fail_votes)
        self.history[f"round {self.round}"]['quest_vote'] = encoded_quest_vote
        
        fail_limit: int = self.num_fails_for_quest[self.round] # 1 or 2
        if fail_votes >= fail_limit:  # Mission failed                        
            self.cum_quest_fails += 1  # cumulative fails

            if self.cum_quest_fails == 3:  # after 3 fails Evil wins
                self.done = True  # XXXXXXXX END OF THE GAME XXXXXXXXXXXX
                self.good_victory = False

                if verbose:
                    print(f"ROUND #{self.round}. Phase: (2, Quest Voting)")
                    print(f'Quest Team: {self.quest_team}')
                    print(f"Quest failed. {fail_votes} player(s) voted to Fail.")
                    print(f'== The end. 3 failed Quests. Evil wins 😈 ')
                    print("=========================================")
            else:
                if self.round != 4:
                    self.round += 1
                    self.phase = 0
                if verbose:
                    print(f'Quest Team: {self.quest_team}')
                    print(f"Quest failed. {fail_votes} player(s) voted to Fail. ")
                    print(f"Total number of failed Quests: {self.cum_quest_fails}.")
                    print(f"The next Round: {self.round}. The next Phase: {self.phase}")
                    print("=========================================")
        
        else: # SUCCESS        
            if self.round - self.cum_quest_fails == 2:  # 3 successes?
                self.done = True  # XXXXXXXX END OF THE GAME XXXXXXXXXXXX
                self.good_victory = True
                if verbose:
                    print(f"ROUND #{self.round}. Phase: (2, Quest Voting)")
                    print(f'Quest Team: {self.quest_team}')
                    print(f"Quest succeeded. {fail_votes} player(s) voted to Fail.")
                    print(f"Total number of failed Quests: {self.cum_quest_fails}.")
                    print(f'== The end. 3 successful Quests. Good wins 😇')
                    print("=========================================")

            else:
                if self.round != 4:
                    self.round += 1
                    self.phase = 0
                if verbose:
                    print(f"ROUND #{self.round}. Phase: (2, Quest Voting)")
                    print(f'Quest Team: {self.quest_team}')
                    print(f"Quest succeeded. {fail_votes} player(s) voted to Fail.")
                    print(f"Total number of failed Quests: {self.cum_quest_fails}.")
                    print(f"The next Round: {self.round}. The next Phase: {self.phase}")
                    print("=========================================")

    def assign_roles(self) -> list:
        # evil_players = np.random.choice(self._agent_ids, self.num_evil, replace=False)
        # evil_players
        # roles = {agent_id: 0 for agent_id in self._agent_ids}
        # for evil_players in evil_players:
        #     roles[evil_players] = 1
        
        good_players = np.random.choice(self._agent_ids, self.num_good, replace=False)
        return list(good_players)
        
    def _encoding_number(self, number: int, max_number: int) -> list:
        """
        One-hot encoding
        """
        enc = [0] * max_number
        enc[number] = 1
        return enc
    
    def _encoding_player(self, agent_id: str) -> list:
        """
        One-hot encoding
        """
        player_id = int(agent_id[-1]) - 1
        return self._encoding_number(player_id , self.num_players)
    
    def _encoding_quest_vote(self, fail_vote: int) -> list:
        if fail_vote == 0:
            encoded_quest_vote = [0, 0, 1]
        if fail_vote == 1:
            encoded_quest_vote = [0, 1, 0]
        if fail_vote == 2:
            encoded_quest_vote = [1, 0, 0]
        return encoded_quest_vote

    def _get_start_history(self) -> dict:
        """
        [:5] round
        [5:10] leader
        round   lea1 team1 tv1    lea2  team2 tv2    lea3 team3 tv3   lea4  team4  tv4  lea5  team5 tv5   qv
        10000  01000 11000 11100  00100 01010 11001 00000 00000 00000 00000 00000 00000 00000 00000 00000 001
        01000  01000 11000 11100  00100 01010 11001 00000 00000 00000 00000 00000 00000 00000 00000 00000 001  
        00000  00000 00000 00000  00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 001

        """
        history = {f'round {i}': None for i in range(5) }
        for round in history:
            history[round] = {
                'round_id':  [0] * 5, 
                'attempt 1': [0] * 15,
                'attempt 2': [0] * 15,
                'attempt 3': [0] * 15,
                'attempt 4': [0] * 15,
                'attempt 5': [0] * 15,
                'quest_vote': [0] * 3,
            }

        history['round 0']['round_id'] = self._encoding_number(self.round, 5)
        history['round 0']['attempt 1'][:5] = self._encoding_player(self.leader)
        return history
    
    def _change_leader(self):
        leader_idx = int(self.leader[-1]) - 1
        next_leader_idx = (leader_idx + 1) % self.num_players 
        self.leader = f"player_{next_leader_idx+1}"

    def _convert_dict_2d_array(self, di):
        # Extract the lists (values) from the dictionary
        values = list(di.values())
        return np.array(values)
    
    def _action_to_array(self, action: int) -> list:
        """
        Retruns a Team from a given action.
        """        
        return self.mapping_dict_action_to_team[action]

    # PREDEFINED ACTIONS
    def _get_predifined_evil_action_phase0(self) -> list:
        """Returns a Team """
        # PHASE 0        
        leader_idx = int(self.leader[-1]) - 1  # for example, 0
        if self.round == 0 or self.round == 2:
            possible_teams = [team for action, team in self.mapping_dict_action_to_team.items() if action <= 10]
        else:
            possible_teams = [team for action, team in self.mapping_dict_action_to_team.items() if action > 10 and action <= 20]
        teams_to_choose = []
        for team in possible_teams:
            if team[leader_idx] == 1:
                teams_to_choose.append(team)
        action  = random.choice(teams_to_choose)
        return action
    
    def _get_predefined_evil_action_phase1(self) -> dict:
        """
        {player_id: 1,
        player_id: 0 }
        """
        predefined_action_dict = {}
        evil_in_team = False
        evil_players = [( int(agent_id[-1]) - 1, agent_id) for agent_id in self._agent_ids if agent_id not in self.good_players]
        # if any evil player in Team vote FOR, otherwise AGAINST
        for (player_idx, player) in evil_players:
            if self.quest_team[player_idx] == 1:
                evil_in_team = True
        
        for _, evil_player in evil_players:
            if evil_in_team:
                predefined_action_dict[evil_player] = 1
            else:
                predefined_action_dict[evil_player] = 0
        
        return predefined_action_dict
    
    def _get_predefined_evil_action_phase2(self) -> dict:
        """
        Always vote AGAINST (1)
        """
        predefined_action_dict = {}
        evil_players = [( int(agent_id[-1]) - 1, agent_id) for agent_id in self._agent_ids if agent_id not in self.good_players]
        for _, evil_player in evil_players:
            predefined_action_dict[evil_player] = 1
        return predefined_action_dict
        



