import math
import numpy as np
import random


#HYPERPARAMETERS
REWARD_PER_DISTANCE = 30
MAX_X = 50
MAX_Y = 50
NUM_OF_ACTIONS = 4
NUM_OF_DIRECTIONS = 4
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.8
EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 0.999
MIN_EXPLORATION_RATE = 0.25
GOAL_STATE = (20,4)
GOAL_STATES = [(18,4),(19,4),(20,4),(21,4),(18,5),(19,5),(20,5),(21,5),(18,6),(19,6),(21,6),(20,6)]

#try to load q-table from file, if it can't, create it from scratch
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
    EXPLORATION_RATE = 1.0
    
    
#calculates immediate reward for reaching a state
#collisions and dropping cargo gives a negative reward
#getting closer to the goal gives a reward
#states that are closer to the goal are valued more
def reward_function(prev_state, next_state,has_collided,cargo):
    reward = 0
    if (has_collided):
        reward -= 1000
    if (cargo == False):
        reward -= 1000
    for state in GOAL_STATES:
        if (state[0] == next_state[0] and state[1] == next_state[1]):
            print("GOAL REACHED")
            return 10000, True
    distance_to_goal = math.sqrt((prev_state[0] - GOAL_STATE[0])**2 + (prev_state[1] - GOAL_STATE[1])**2)
    new_distance_to_goal = math.sqrt((next_state[0] - GOAL_STATE[0])**2 + (next_state[1] - GOAL_STATE[1])**2)
    change_in_distance = distance_to_goal - new_distance_to_goal
    reward += change_in_distance * REWARD_PER_DISTANCE
    if (change_in_distance > 0):
        reward += 100 / new_distance_to_goal
    return reward, False
    
#find which action to do at a state
#follows an epsilon greedy strategy, either chooses an action from the policy function or use the q-table to find the best action (that has been found yet)
def q_value_action(state_data):
    global EXPLORATION_RATE
    action_choice = -1
    state = state_to_index(state_data)
    choice = random.randint(1,10)
    if (choice < EXPLORATION_RATE * 10):
        action_choice = policy()
    else:
        print("The greedy has a choice of : " , Q_table[state[0]][state[1]][state[2]])
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
    reward,goal = reward_function(state,next_state,has_collided,cargo)
    Q_table[state[0]][state[1]][state[2]][action_index] = Q_table[state[0]][state[1]][state[2]][action_index] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * Q_table[state[0]][state[1]][state[2]].max() - Q_table[state[0]][state[1]][state[2]][action_index])
    return goal

#this policy favours foward movement to make sure the robot isnt stuck going fowards and backwards, or left and right
def policy():
   choice = random.randint(1,22) 
   if (choice <= 9 ):
       return 0
   elif (choice <= 14):
       return 1
   elif (choice <= 19):
       return 2
   else:
       return 3
       
#saves q-table and exploration rate in a file for the future simulation runs    
def save_q_table(path):
    np.save(path,Q_table)
    f = open("../lib/exploration_rate.txt","w")
    f.write(str(EXPLORATION_RATE))
    

#translates the state data into indexes for the q-table
def state_to_index(state_data):
    state_indexes = [0,0,0]
    state_indexes[0] = int(np.round(state_data.position_x))
    state_indexes[1] = int(np.round(state_data.position_y))
    state_indexes[2] = heading_to_index(state_data.heading)
    if (state_indexes[2] == -1):
        print("Invalid heading in state")
    return state_indexes

#converts the bearing of the robot to an index for the q-table
#returns -1 to print a statement to alert that there is an invalid bearing in the state
def heading_to_index(h):
    if h is None or h != h:
        return -1
    if h > 315 or h <= 45:
        return 0        
    elif h >= 45 and h <= 135:
        return 1
    elif h >= 135 and h <= 225:
        return 2
    elif h >= 225 and h <= 315:
        return 3
    else:
        return -1
                
        
        
        
        
        
        
        