!wget https://github.com/andrew-veriga/DL/raw/master/assign.zip
!unzip -u assign.zip
!rm assign.zip

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from rlglue.rl_glue import RLGlue
import main_agent
import ten_arm_env
import test_env

def argmax(q_values):
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
	    if (q_values[i] < top_value):
	        continue

	    if (q_values[i] > top_value):
	        top_value = q_values[i]
	        ties = []

	    ties.append(i)

    return np.random.choice(ties)



class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        number_actions = self.arm_count[self.last_action] + 1;
        current_value = self.q_values[self.last_action];
        new_value = current_value + (1 / number_actions) * (reward - current_value);

        self.arm_count[self.last_action] = number_actions;
        self.q_values[self.last_action] = new_value;

        current_action = argmax(self.q_values);
        self.last_action = current_action

        return current_action



class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        number_actions = self.arm_count[self.last_action] + 1;
        current_value = self.q_values[self.last_action];
        new_value = current_value + (1 / number_actions) * (reward - current_value);

        self.arm_count[self.last_action] = number_actions;
        self.q_values[self.last_action] = new_value;

        current_action = np.random.randint(len(self.arm_count)) if np.random.random() < self.epsilon else argmax(self.q_values);
        self.last_action =  current_action;

        return current_action



class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        number_actions = self.arm_count[self.last_action] + 1;
        current_value = self.q_values[self.last_action];
        new_value = current_value + self.step_size * (reward - current_value);

        self.arm_count[self.last_action] = number_actions;
        self.q_values[self.last_action] = new_value;

        current_action = argmax(self.q_values);
        self.last_action = current_action
        
        current_action = np.random.randint(len(self.arm_count)) if np.random.random() < self.epsilon else argmax(self.q_values);

        self.last_action = current_action
        
        return current_action
