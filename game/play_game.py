import random

PHASE_NAMES = {
    0: 'Team Building',
    1: 'Team Voting',
    2: 'Quest Voting',
}


def play_game(env, config: dict, n_humans=1):
    """
    n_humans: numer of human players
    Example config:
        config = {
            0: mappo_good_policy,
            1: mappo_evil_policy,
        }
    
    policies = {
        player_1: mappo_good_policy,
        player_2: mappo_evil_policy,
        player_3: naive_evil_policy,
        ...
    }    
    """
    mapping_dict_action = {
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
           24 : 1,  # Vote to Fail in Quest
        }

    def _convert_human_input_to_action(s: str) -> int:
        """
        '1, 3' - > [1, 3] -> 2
        """
        li = s.split(',')
        li = [int(i)-1 for i in li]
        team = [0] * 5
        for el in li:
            team[el] = 1
        
        for action, team_enc in mapping_dict_action.items():
            if team_enc == team:
                chosen_action = action
                break
        return chosen_action

    stat = {}
    # Reset a Game
    obs, share_obs, available_actions, info = env.reset()
    done = False

    # Set Up human players
    human_players = []
    computer_players = [agent_id for agent_id in env._agent_ids]
    for i in range(n_humans):
        human_player = random.choice(computer_players)
        computer_players.remove(human_player)
        human_players.append(human_player)

    policies = {
        agent_id: config[0] if agent_id in env.good_players else config[1] for agent_id in computer_players
    }

    counter_step = 0
    
    print('-- Start of the Game --')
    for human_player in human_players:
        role = '😇' if human_player in env.good_players else '😈'
        print(f'You are {human_player}. You are {role}')
        if role == '😈':
            another_evil = [evil_player for evil_player in env.evil_players if evil_player != human_player][0]
            print(f'Another Evil player is {another_evil}')

    while not done:
        print("\n")
        print(f'=== Round: {env.round+1}, Phase: {PHASE_NAMES[env.phase]}')
        action_dict = {}
        for agent_id in computer_players:
            # compute an Action and put it in the dictionary
            values, actions, action_log_probs, action_probs = policies[agent_id].get_actions(
                share_obs[agent_id], obs[agent_id], available_actions[agent_id])
            action = actions.item()
            action_dict[agent_id] = action
        
        for human in human_players:
            if env.phase == 0:
                if env.leader == human:
                    print(f'You are a Leader. Choose a Team of {env.num_players_for_quest[env.round]} players.')
                    chosen_team_str = input("Input player's indicies. For example: 1, 3. Enter: ")
                    chosen_action = _convert_human_input_to_action(chosen_team_str)
                    action_dict[human] = chosen_action
                else:
                    for act in action_dict.values():
                        if act != 0:
                            print(f'{env.leader} chose Team: {mapping_dict_action[act]}')
            elif env.phase == 1:
                print(f'Vote for the proposed Team. **Attempt** {env.voting_attempt}.')
                team_vote = int(input("Enter 1 (support) or 0 (against): "))
                chosen_action  = 21 if team_vote == 1 else 22
                action_dict[human] = chosen_action

            elif env.phase == 2:
                if env._is_in_quest_team(human):
                    print('Vote for the Quest: 1 - Fail, 0 - Succeed')
                    quest_vote = int(input("Enter 1 or 0: "))
                    chosen_action  = 24 if quest_vote == 1 else 23
                    action_dict[human] = chosen_action

        if env.phase == 1:
            print('Votes:')
            for agent_id in env._agent_ids:
                print(f'{agent_id}: {mapping_dict_action[action_dict[agent_id]]}')
        if env.phase == 2:
            num_fails = sum([ mapping_dict_action[act] for act in action_dict.values() if act != 0 ])
            
            print(f'Quest Vote Results (Team: {env.quest_team}): {num_fails} fails.')

        counter_step += 1

        # Step environment
        obs, share_obs, available_actions, rewards, done, info = env.step(action_dict)      
        
    # Game end
    print(f'The game ended in {counter_step} turns.')
    print(f'Number of successful Quests: {env.round - env.cum_quest_fails}')
    if env.good_victory:
        print(f'😇 Good Team won 😇') 
    else:
        print(f'😈 Evil Team won 😈') 

    # Collecting statistics
    stat['number of turns'] = counter_step
    stat['good victory'] = 1 if env.good_victory else 0
    stat['number of rounds'] = env.round + 1

    return stat      
        
