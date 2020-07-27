from A2C import A2CAgent
import numpy as np
import random
import torch
from rubix import *
from torch import nn

from matplotlib import pyplot as plt





def performAction(cube, action):
    count = 0
    for state in cube.get_successors(cube.F):
        if count == action:
            return state
        count += 1
    return None

n_episodes = 10000
gamma = 0.99
num_actions = 18

eprewards = []
losses = []

def one_hot(state):
    return np.eye(6)[state]

cube = Cube(3)

model = A2CAgent(54, num_actions, 256)
optimizer = torch.optim.Adam(model.parameters(), 0.0005)

for ep in range(n_episodes):
    cube = Cube(3)
    state = cube.scramble(cube.F, 5)
    epreward = 0
    maxeplen = 30   
    eplen = 0
    # Each episode
    done = False
    probs = []
    rewards = []
    values = []
    while not done:
        # print(one_hot(state.flatten()).shape)
        value, policy = model(torch.FloatTensor(one_hot(state.flatten())[None,None,:,:]))
        action = np.random.choice(num_actions, p = policy.detach().numpy())
        probs.append(-torch.log(policy[action]))

        next_state = performAction(cube, action)

        moves_to_solve = len(cube.astar(next_state, cube.h))

        solved_cube = cube.is_goal_state(next_state)
        if solved_cube:
            reward = 1
            done = True
        elif eplen > maxeplen:
            reward = -moves_to_solve
            done = True
        else:
            reward = -moves_to_solve
            done = False



        epreward += reward

        rewards.append(reward)

        values.append(value)

        state = next_state
        eplen += 1


    optimizer.zero_grad()
    loss = model.loss(torch.stack(probs), torch.FloatTensor(rewards), torch.stack(values))

    print("Episode {}: Total Reward: {}, Loss: {}".format(ep,epreward,loss.item()))
    eprewards.append(epreward)
    losses.append(loss.item())

    
    loss.backward()
    optimizer.step()

        
    #     next_state, reward, done, info = env.step(action)
    #     epreward += reward
    #     replayMemory.add(state, action, reward, next_state, done)

    #     if (len(replayMemory) >= batch_size):
    #         samples = replayMemory.sample(min(len(replayMemory), batch_size))
    #         y = [sample["reward"] if sample["done"] else sample["reward"] + \
    #             gamma * torch.max(model(torch.tensor(sample["next_state"]))) for sample in samples]
            
    #         optimizer.zero_grad()

    #         states = torch.tensor([sample["state"] for sample in samples])
    #         pred = model(states)
    #         actions = torch.tensor([[sample["action"]] for sample in samples])
    #         qs = torch.squeeze(torch.gather(pred, 1, actions))
    #         y_batch = torch.tensor(y)

            
    #         l = loss(qs, y_batch)
            
    #         l.backward()
    #         optimizer.step()

    #     state = next_state
    # print(epreward)
    if (ep % 1000 == 0):
        print("---------------------------------Saving---------------------------------")
        torch.save(model.state_dict(), './model.pt')


np.savez('episode_rewards.npz', payload=eprewards)
np.savez('losses.npz', payload=losses)
