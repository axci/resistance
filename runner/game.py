import torch


def simulate_game(env, config: dict, verbose=False):
    """
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
    stat = {}
    # Reset a Game
    obs, share_obs, available_actions, info = env.reset()
    done = False

    policies = {
        agent_id: config[0] if agent_id in env.good_players else config[1] for agent_id in env._agent_ids 
    }

    counter_step = 0
    
    if verbose:
        print('-- Start of the Game --')
        print(f'Good Team: {env.good_players}')
        print(f'Evil Team: {env.evil_players}')

    while not done:
        if verbose:
            print('===================')
            print(f'Round: {env.round}, Phase: {env.phase}, Vote attempt {env.voting_attempt}')
        action_dict = {}
        for agent_id in env._agent_ids:
            # compute action and put it in the dictionary
            values, actions, action_log_probs, action_probs = policies[agent_id].get_actions(
                share_obs[agent_id], obs[agent_id], available_actions[agent_id])
            action = actions.item()
            action_dict[agent_id] = action


            if verbose:
                if action != 0:
                    if agent_id in env.good_players:
                        print(f'{agent_id} 😇: {env.mapping_dict_action_to_team[action]} ({torch.max(action_probs):.2f})')
                    else:
                        print(f'{agent_id} 😈: {env.mapping_dict_action_to_team[action]} ({action_probs})')

        
        
        if verbose:
            if env.phase == 2:
                number_of_fails = sum([env.mapping_dict_action_to_team[action] for action in list(action_dict.values()) if action != 0])
                total_number_fails = env.cum_quest_fails
                total_number_success = env.round - env.cum_quest_fails    
                if number_of_fails > 0:
                    total_number_fails += 1
                else:
                    total_number_success += 1
                print(f' ---> Round {env.round}: 😇 {total_number_success} : {total_number_fails} 😈')
        counter_step += 1
        # Step environment
        obs, share_obs, available_actions, rewards, done, info = env.step(action_dict)
        if verbose:
            print('Rewards')
            for agent_id, action in action_dict.items():
                if action != 0:
                    if agent_id in env.good_players:
                        print(f'{agent_id} 😇: {rewards[agent_id]}')
                    else:
                        print(f'{agent_id} 😈: {rewards[agent_id]}')

        


        
    
    if verbose:
        print(f'The game ended in {counter_step} turns.')
        
        print(f'Final Rewards: ')
        for agent_id in env._agent_ids:
            if agent_id in env.good_players:
                print(f'{agent_id} 😇: {rewards[agent_id]}')
            else:
                print(f'{agent_id} 😈: {rewards[agent_id]}')
            
        print(f'Number of successful Quests: {env.round + 1 - env.cum_quest_fails}')
        if env.good_victory:
            print(f'😇 Good Team won 😇')
        else:
            print(f'😈 Evil Team won 😈')

    # Collecting statistics
    stat['number of turns'] = counter_step
    stat['good victory'] = 1 if env.good_victory else 0
    stat['number of rounds'] = env.round + 1

    return stat
        
        
