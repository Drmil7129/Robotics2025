import math
import numpy as np
import random


#TO-DO:
#Make a separate ruleset for testing


#HYPERPARAMETERS
REWARD_PER_DISTANCE = 1
MAX_X = 50
MAX_Y = 50
NUM_OF_ACTIONS = 4
NUM_OF_DIRECTIONS = 4
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5
EXPLORATION_RATE = 0
EXPLORATION_RATE_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.1
MAX_TIMESTEPS = 300
MAX_DISCOUNT_EXPONENT = 10
GOAL_STATE = (20,2)

#try to load q-table from file, if it can't, creaste it from scratch
try:
    Q_table = np.load("../lib/q_table.npy")

except:
    Q_table = np.zeros((MAX_X,MAX_Y,NUM_OF_DIRECTIONS,NUM_OF_ACTIONS))

#try to load exploration rate value from file, if not set it to default value
try: 
    f = open("../lib/exploration_rate.txt","r")
    EXPLORATION_RATE = float(f.read())
    print("exploration rate is ", EXPLORATION_RATE)
except:
    print("Couldnt load")
    EXPLORATION_RATE = 0.5
    
    
#calculates immediate reward for reaching a state
def reward_function(prev_distance, next_distance,has_collided,cargo):
    reward = 0
    if (has_collided):
        reward -= 1000
    if (cargo == False):
        reward -= 1000
    distance_to_goal = math.sqrt((prev_distance[0] - GOAL_STATE[0])**2 + (prev_distance[1] - GOAL_STATE[1])**2)
    new_distance_to_goal = math.sqrt((next_distance[0] - GOAL_STATE[0])**2 + (next_distance[1] - GOAL_STATE[1])**2)
    change_in_distance = distance_to_goal - new_distance_to_goal
    reward += change_in_distance * REWARD_PER_DISTANCE
    return reward
    
#find which action to do at a state
def q_value_action(state_data):
    global EXPLORATION_RATE
    action_choice = -1
    state = state_to_index(state_data)
    choice = random.randint(0,10)
    if (choice < EXPLORATION_RATE * 10):
        action_choice = policy()
    else:
        action_choice = np.argmax(Q_table[state[0]][state[1]][state[2]])
    if (EXPLORATION_RATE > MIN_EXPLORATION_RATE):
        EXPLORATION_RATE = EXPLORATION_RATE * EXPLORATION_RATE_DECAY
    print("State: ", state, " undergos action: ", action_choice)
    return action_choice    
 
#updates q-value using the q-value update rule  
def q_value_update(state_data,next_state_data,action,has_collided,cargo):
    global Q_table
    state = state_to_index(state_data)
    next_state = state_to_index(next_state_data)
    action_index = action
    Q_table[state[0]][state[1]][state[2]][action_index] = Q_table[state[0]][state[1]][state[2]][action_index] + LEARNING_RATE * (reward_function(state,next_state,has_collided,cargo) + DISCOUNT_FACTOR * Q_table[state[0]][state[1]][state[2]].max() - Q_table[state[0]][state[1]][state[2]][action_index])
    #print("The value for state ", state, " is ", Q_table[state[0]][state[1]][state[2]][action_index])

def policy():
   choice = random.randint(0,10) 
   if (choice < 6):
       return 0
   elif (choice > 5 and choice < 8):
       return 1
   elif (choice > 7 and choice < 9):
       return 2
   else:
       return 3
    
def save_q_table(path):
    np.save(path,Q_table)
    f = open("../lib/exploration_rate.txt","w")
    f.write(str(EXPLORATION_RATE))
    

#translates the state data into indexes for the q-table
def state_to_index(state_data):
    state_indexes = [0,0,0]
    state_indexes[0] = int(np.round(state_data.position_x + 24))
    state_indexes[1] = int(np.round(state_data.position_y + 24))
    state_indexes[2] = heading_to_index(state_data.heading)
    return state_indexes


def heading_to_index(h):
    if h is None or h != h:  # Check for None or NaN
        return 0
    if h > 315 or h <= 45:
        return 0        
    elif h >= 45 and h <= 135:
        return 1
    elif h >= 135 and h <= 225:
        return 2
    elif h >= 225 and h <= 315:
        return 3
    else:
        return 0  # Default case
                
        
        
        
        
        
        
        