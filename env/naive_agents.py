import numpy as np
import random

class NaiveEvilPolicy():
    def __init__(self):
        super().__init__()
        self.team_2 = {
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
        }
        self.team_3 = {
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
        }

    def get_actions(self, share_obs: np.ndarray, obs: np.ndarray, available_actions):
        player_idx = np.where(   np.array(obs[:5]) == 1)[0][0]
        
        if available_actions[1] == 1: # if I am leader and the Phase is 0 and team size is 2
            possible_teams = [(action, team) for action, team in self.team_2.items()]
            actions_to_choose = []
            for (action, team) in possible_teams:
                if team[player_idx] == 1:
                    actions_to_choose.append(action)
            chosen_action  = random.choice(actions_to_choose)
        
        elif available_actions[11] == 1: # if I am leader and the Phase is 0 and team size is 3
            possible_teams = [(action, team) for action, team in self.team_3.items()]
            actions_to_choose = []
            for (action, team) in possible_teams:
                if team[player_idx] == 1:
                    actions_to_choose.append(action)
            chosen_action  = random.choice(actions_to_choose)
        
        elif available_actions[21] == 1: # Team Voting
            evil_in_team = False
            evil_players = obs[-5:]
            round_0_hist = obs[5:85]
            round_1_hist = obs[88:168]
            round_2_hist = obs[171:251]
            round_3_hist = obs[254:334]
            round_4_hist = obs[337:417]
            hist = [round_4_hist, round_3_hist, round_2_hist, round_1_hist, round_0_hist]
            for round_hist in hist:
                if sum(round_hist) != 0:
                    team_to_vote_for = self._find_first_5_elements_with_one(round_hist)
                    break
            chosen_action = 22
            #print('team to vote for', team_to_vote_for)
            #print('knowledge', evil_players)
            for i, player in enumerate(evil_players): # [0, 1, 1, 0, 0]
                if player == 1:
                    if player == team_to_vote_for[i]:
                        chosen_action = 21
        
        elif available_actions[23] == 1:  # Quest Voting
            chosen_action = 24
        else:
            chosen_action = 0

        values = None
        actions = np.array(chosen_action)
        action_log_probs = None
        action_probs = None
        return values, actions, action_log_probs, action_probs
    
    def _find_first_5_elements_with_one(self, lst):
    # Loop through the list in reversed order in chunks of 5 elements
        for i in range(len(lst) - 1, 3, -5):
            # Get the current chunk of 5 elements
            chunk = lst[i-4:i+1]
            
            # Check if there is at least one `1` in the current chunk
            if 1 in chunk:
                return chunk
        
        # If no chunk with a `1` is found, return an empty list or None
        return []
    
