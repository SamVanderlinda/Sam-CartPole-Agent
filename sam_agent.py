# Cart the uses BASIC machine learning to hold pole
# vetically as long as possible
# Physics within the cart-pole game will dictate movement of the pole
# based on movements from the cart 
# nueral network and calculations are going on in the background

import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')
env._max_episode_steps = 300


# Runs the agent for 200 actions before training
def random_agent():
    
    print("Before Training:")
    env.reset() # must reset evironment before you use it
    test = 0
    done = False
    
    for _ in range(300):
        env.render()
        # within the action space give a bunch of random movements
        done = env.step(env.action_space.sample())
        test += 1
        
        if test > 200 and done:
            env.close()
            break

#Trains and runs the agent
def training_running():
    bestLength = 0 
    episode_lengths = []
    # start with four 0's for weights
    best_weights = np.zeros(4)

    # Train for 10000 episodes
    
    for i in range(100):
        # start with random weights every 100 episode to explore better
        # agent
        new_weights = np.random.uniform(-2.0, 2.0, 4)
        length = []

        # Train for 100 episodes to optimize weights
        for j in range(100):
            # obersvation is an array where: 
            # [cart postition, cart velocity, pole angle, pole velocity at tip]
            observation = env.reset() 
            done = False
            count = 0

            while not done:
                count += 1
                # move right if dot product is positive, move left if negative 
                action = 1 if np.dot(observation, new_weights) > 0 else 0
                # perform that action in the environment
                # observation of state, reward 1 point the pole is vertical
                # done flags when reaching 300 steps, _ debug info
                observation, reward, done, _ = env.step(action)
                
                if done:
                    break
            
            length.append(count)
        
        # this will tell us how well these weights performed
        average_length = float(sum(length) / len(length))

        # updates the best new weights
        if average_length > bestLength:
            bestLength = average_length
            best_weights = new_weights
        
        episode_lengths.append(average_length)
        
        #Every 100 episodes of trainging report the best length
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
