import math
import numpy as np

REWARD_PER_DISTANCE = 1
MAX_X = 50
MAX_Y = 50
NUM_OF_ACTIONS = 6
NUM_OF_DIRECTIONS = 4
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5
Q_table = np.zeros(MAX_X,MAX_Y,NUM_OF_DIRECTION,NUM_OF_ACTIONS)



#calculates immediate reward for reaching a state
def reward_function(prev_distance, next_distance):
    change_in_distance = math.sqrt((prev_distance[0] - next_distance[0])**2 + (prev_distance[1] - next_distance[1])**2)
    reward = change_in_distance * REWARD_PER_DISTANCE
    return reward
    
#find which action to do at a state, this favours exploitation over exploration, due to the large Q-table size
def q_value_action(state):
    direction = get_direction(state[2])
    choice = random.randint(0,10)
    if (choice > 7):
        action_choice = random.randint(0,NUM_OF_ACTIONS-1)
    else:
        action_choice = np.argmax(Q_table[state[0]][state[1]][direction])
    return action_choice    
    
    
def get_direction(n):
    return (round(n/90) % 4)