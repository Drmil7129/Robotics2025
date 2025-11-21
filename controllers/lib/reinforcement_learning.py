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

Q_table = np.zeros((MAX_X,MAX_Y,NUM_OF_DIRECTIONS,NUM_OF_ACTIONS))



#calculates immediate reward for reaching a state
def reward_function(prev_distance, next_distance):
    #we need code here that detects if the robot has crashed into an obstacle, or if the package has fallen off
    #we need IR sensors and/or cameras to detect this
    distance_to_goal = math.sqrt((prev_distance[0] - GOAL_STATE[0])**2 + (prev_distance[1] - GOAL_STATE[1])**2)
    print("Distance to goal is ", distance_to_goal)
    new_distance_to_goal = math.sqrt((next_distance[0] - GOAL_STATE[0])**2 + (next_distance[1] - GOAL_STATE[1])**2)
    print("New distance to goal is ", new_distance_to_goal)
    change_in_distance = distance_to_goal - new_distance_to_goal
    reward = change_in_distance * REWARD_PER_DISTANCE
    return reward
    
#find which action to do at a state, this favours exploitation over exploration, due to the large Q-table size
def q_value_action(state):
    choice = random.randint(0,10)
    if (choice > EXPLORATION_RATE  ):
        action_choice = random.randint(0,NUM_OF_ACTIONS-1)
        print("The action choice is ", action_choice)
    else:
        action_choice = np.argmax(Q_table[state[0]][state[1]][state[2]])
        print("The action choice is ", action_choice)
    return action_choice    
 
#updates q-value using the q-value update rule  
def q_value_update(state,next_state,action):
    print("State is ",state)
    print("Next state is ", next_state)
    action_index = action
    Q_table[state[0]][state[1]][state[2]][action_index] = Q_table[state[0]][state[1]][state[2]][action_index] + LEARNING_RATE * (reward_function(state,next_state) + DISCOUNT_FACTOR * Q_table[state[0]][state[1]][state[2]].max() - Q_table[state[0]][state[1]][state[2]][action_index])
    print("The value of the reward ",Q_table[state[0]][state[1]][state[2]][action_index] )

#finds how good it is being in this state, by adding the immediate reward and the future discount total reward wehn following the policy    
def value_function(current_state,previous_state,discount_exponent):
    if (discount_exponent == MAX_DISCOUNT_EXPONENT):
        return 0
    else:
        reward = reward_function(current_state,previous_state)
        #action = getActionFromPolicy
        return reward
    


#converts action representation to an index for the q-table
def get_index_from_action(action):
    pass