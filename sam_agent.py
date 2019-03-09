#A game to balance a pole as long as possible
#This code illustrates simple machine learning before and after training 
#First render is a random agent, performing 200 random micromovements 
#The agent is then trained without a render
#Second render is a the trained agent

import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')
env._max_episode_steps = 300

#Runs the agent for 200 actions before training
def random_agent():
    print("Before Training:")
    env.reset()
    test = 0
    done = False
    for _ in range(1000):
        env.render()
        done = env.step(env.action_space.sample())
        
        test += 1
        if test > 200 and done:
            env.close()
            break

#Trains and runs the agent
def training_running():
    
    bestLength = 0 
    episode_lengths = []
    best_weights = np.zeros(4)

    for i in range(100):
        new_weights = np.random.uniform(-2.0, 2.0, 4)
        length = []

        for j in range(100):
            observation = env.reset()
            done = False
            count = 0

            while not done:
                count += 1
                action = 1 if np.dot(observation, new_weights) > 0 else 0
                observation, reward, done, _ = env.step(action)
                if done:
                    break
            length.append(count)
        average_length = float(sum(length) / len(length))

        if average_length > bestLength:
            bestLength = average_length
            best_weights = new_weights
        episode_lengths.append(average_length)
        if i % 10 == 0:
            print('best length is ', bestLength)
    done = False
    count = 0

    #Execute the optimal agent
    observation = env.reset()
    print("After Training:")
    while not done:
        env.render()
        count += 1

        action = 1 if np.dot(observation, best_weights) > 0 else 0
        observation, reward, done, _ = env.step(action)
        if done:
            env.close()
            break

random_agent()
training_running()