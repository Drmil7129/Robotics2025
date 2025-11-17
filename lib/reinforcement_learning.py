import math
import numpy as np
import random


#TO-DO:
#Integrate Q-learning into main loop
#Define the actions for robot
#Timestep configuration
#Integrate policy
#Make a separate ruleset for testing


#HYPERPARAMETERS
REWARD_PER_DISTANCE = 1
MAX_X = 50
MAX_Y = 50
NUM_OF_ACTIONS = 4
NUM_OF_DIRECTIONS = 4
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5
EXPLORATION_RATE = 0.5
EXPLORATION_RATE_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.1
MAX_TIMESTEPS = 300
MAX_DISCOUNT_EXPONENT = 10
NUM_OF_EPISODES = 10000
GOAL_STATE = (21,47)

Q_table = np.zeros(MAX_X * MAX_Y * NUM_OF_DIRECTION,NUM_OF_ACTIONS)



#calculates immediate reward for reaching a state
def reward_function(prev_distance, next_distance):
    #we need code here that detects if the robot has crashed into an obstacle, or if the package has fallen off
    #we need IR sensors and/or cameras to detect this
    distance_to_goal = math.sqrt((prev_distance[0] - GOAL_STATE[0])**2 + (prev_distance[1] - GOAL_STATE[1])**2)
    new_distance_to_goal = math.sqrt((next_distance[0] - GOAL_STATE[0])**2 + (next_distance[1] - GOAL_STATE[1])**2)
    change_in_distance = distance_to_goal - new_distance_to_goal
    reward = change_in_distance * REWARD_PER_DISTANCE
    return reward
    
#find which action to do at a state, this favours exploitation over exploration, due to the large Q-table size
def q_value_action(state):
    choice = random.randint(0,10)
    if (choice > EXPLORATION_RATE  ):
        action_choice = random.randint(0,NUM_OF_ACTIONS-1)
    else:
        action_choice = np.argmax(Q_table[get_index_from_state(state)])
    return action_choice    
 
#updates q-value using the q-value update rule  
def q_value_update(state,action,previous_state):
    table_index = get_index_from_state(state)
    action_index = get_index_from_action(action)
    Q_table[table_index][action_index] = Q_table_index[table_index][action_index] + 
    LEARNING_RATE * (reward_function(previous_state,state) + 
    DISCOUNT_FACTOR * Q_table[table_index].max() -
    Q_table[table_index][action_index])
    

#finds how good it is being in this state, by adding the immediate reward and the future discount total reward wehn following the policy    
def value_function(current_state,previous_state,discount_exponent):
    if (disocunt_exponent == MAX_DISCOUNT_EXPONENT):
        return 0
    else:
        reward = reward_function(current_state,previous_state)
        #action = getActionFromPolicy
        return reward
    

#converts state representation to an index for the q-table
def get_index_from_state(state):
    pass
    
#converts action representation to an index for the q-table
def get_index_from_action(action):
    pass