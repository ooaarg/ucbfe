import numpy as np
import matplotlib.pyplot as plt
import pickle


"""
Sort arms by method

"""

def ucb(delta, values, counts, imps, n_pos, prices):
  # If prob is not in epsilon, do exploitation of best arm so far
  ucb_values = values + np.sqrt(delta * np.log(10000)/ imps)
  indices = np.argsort(ucb_values * prices)
  return indices[: :-1]

def base(delta, values, counts, imps, n_pos, prices):
  # If prob is not in epsilon, do exploitation of best arm so far
  indices = np.argsort(values * prices)
  return indices[: :-1]

def optimal(true_bandit_probs, prices):
  # If prob is not in epsilon, do exploitation of best arm so far
  indices = np.argsort(true_bandit_probs * prices)
  return indices[: :-1]


def main_loop(n_arms, true_bandit_probs, prices, positions, delta, n_pos, n_iterations, iteration):
    imps_ucb = 0.3 + np.zeros(n_arms)
    counts_ucb = 1 + np.zeros(n_arms)  # How many times each arm was played
    values_ucb = np.zeros(n_arms)  # Estimated reward probability of each arm

    rewards_ucb = np.zeros([n_iterations, n_arms])  # Reward history
    selected_arms_ucb = np.zeros([n_iterations, n_arms])  # Arm selection history


    imps_base = 0.3 + np.zeros(n_arms)
    counts_base = 1 + np.zeros(n_arms)  # How many times each arm was played
    values_base = np.zeros(n_arms)  # Estimated reward probability of each arm

    rewards_base = np.zeros([n_iterations, n_arms])  # Reward history
    selected_arms_base = np.zeros([n_iterations, n_arms])  # Arm selection history


    imps_opt = 0.3 + np.zeros(n_arms)
    counts_opt = 1 + np.zeros(n_arms)  # How many times each arm was played
    values_opt = 1 + np.zeros(n_arms)  # Estimated reward probability of each arm

    rewards_opt = np.zeros([n_iterations, n_arms])  # Reward history
    selected_arms_opt = np.zeros([n_iterations, n_arms])  # Arm selection history
    for i in range(n_iterations):
        arms_ucb = ucb(delta, values_ucb, counts_ucb, imps_ucb, n_pos, prices)
        arms_base = base(delta, values_base, counts_base, imps_base, n_pos, prices)
        arms_opt = optimal(true_bandit_probs, prices)

        #will we get a win (1) or not (0)?
        for pos in range(n_pos):

            per = np.random.rand()
            
            arm_ucb = arms_ucb[pos]
            reward_ucb = per < positions[pos] * true_bandit_probs[arm_ucb] #will we get a win or not? np.random.binomial(1, positions[pos] * true_bandit_probs[arm_ucb])#
            rewards_ucb[i, arm_ucb] = reward_ucb
            #updating counters
            selected_arms_ucb[i, arm_ucb] = positions[pos]
            counts_ucb[arm_ucb] += 1
            imps_ucb[arm_ucb] += positions[pos]
            values_ucb[arm_ucb] = np.sum(rewards_ucb[:,arm_ucb]) / imps_ucb[arm_ucb]


            arm_base = arms_base[pos]
            reward_base = per < positions[pos] * true_bandit_probs[arm_base] #will we get a win or not? np.random.binomial(1, positions[pos] * true_bandit_probs[arm_base])#
            rewards_base[i, arm_base] = reward_base
            #updating counters
            selected_arms_base[i, arm_base] = positions[pos]
            counts_base[arm_base] += 1
            imps_base[arm_base] += positions[pos]
            values_base[arm_base] = np.sum(rewards_base[:,arm_base]) / imps_base[arm_base]


            arm_opt = arms_opt[pos]
            reward_opt = per < positions[pos] * true_bandit_probs[arm_opt] #np.random.binomial(1, positions[pos] * true_bandit_probs[arm_opt])# #will we get a win or not?
            rewards_opt[i, arm_opt] = reward_opt
            #updating counters
            selected_arms_opt[i, arm_opt] = positions[pos]
            counts_opt[arm_opt] += 1
            imps_opt[arm_opt] += positions[pos]
            values_opt[arm_opt] = np.sum(rewards_opt[:,arm_opt]) / imps_opt[arm_opt]

    # Define the path to save the data
    if (prices == np.array([1, 1, 1, 1, 1])).all():
        save_path = f'datav1/iteration_{iteration}.pkl'
    elif (prices == true_bandit_probs * 10).all():
        save_path = f'datav3/iteration_{iteration}.pkl'
    elif (prices  == 1 / np.array([0.45, 0.35, 0.25, 0.15, 0.05]) ** 1.25).all():
        save_path = f'datav4/iteration_{iteration}.pkl'
    else:
        save_path = f'datav5/iteration_{iteration}.pkl'


    e_reward_ucb = np.sum(selected_arms_ucb * true_bandit_probs * prices, axis = 1)
    e_reward_base = np.sum(selected_arms_base * true_bandit_probs * prices, axis = 1)
    e_reward_opt = np.sum(selected_arms_opt * true_bandit_probs * prices, axis = 1)

    rew_ucb = np.sum(rewards_ucb, axis = 1)
    rew_opt = np.sum(rewards_opt, axis=1)
    rew_base = np.sum(rewards_base, axis=1)

    # Create a dictionary to hold the data
    data_to_save = {
        'e_reward_ucb': e_reward_ucb,
        'e_reward_base': e_reward_base,
        'e_reward_opt': e_reward_opt,
        'rew_ucb': rew_ucb,
        'rew_opt': rew_opt,
        'rew_base': rew_base
    }
    # Open the file in write binary mode
    with open(save_path, 'wb') as file:
        # Use pickle.dump() to save the data
        pickle.dump(data_to_save, file)