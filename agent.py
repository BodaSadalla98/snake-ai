from collections import deque
from sys import path
from numpy.lib.function_base import select
import torch
import random
import numpy as np
import  os
from game import SnakeGame, Direction,Point, BLOCK_SIZE
from model import  Trainer, Liner_Qnet
from plot import  plot
MAX_MEM = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.eps = 0 # randomness
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEM)

        if not os.path.exists('model'):
            self.model = Liner_Qnet(11,256,3)
        else:
            self.model = torch.load('model/model.pth')
            self.model.train()
   
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)

        

    def get_state(self, game):

        head = game.head

        ptn_left = Point(head.x - BLOCK_SIZE, head.y)
        ptn_right = Point(head.x + BLOCK_SIZE, head.y)
        ptn_up = Point(head.x, head.y - BLOCK_SIZE)
        ptn_down = Point(head.x, head.y + BLOCK_SIZE)

        



        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        food_up = game.food.y < head.y
        food_down = game.food.y > head.y
        food_right = game.food.x > head.x
        food_left = game.food.x < head.x

        
        danger_straight = (game.is_collision(ptn_right) and dir_r) or(game.is_collision(ptn_left) and dir_l) or (game.is_collision(ptn_up) and dir_u) or (game.is_collision(ptn_down) and dir_d)
        danger_right = (game.is_collision(ptn_down) and dir_r) or(game.is_collision(ptn_up) and dir_l) or (game.is_collision(ptn_right) and dir_u) or (game.is_collision(ptn_left) and dir_d)
        danger_left = (game.is_collision(ptn_up) and dir_r) or(game.is_collision(ptn_down) and dir_l) or (game.is_collision(ptn_left) and dir_u) or (game.is_collision(ptn_right) and dir_d)


        state =  [danger_straight,
                danger_right,
                danger_left,
                dir_l,
                dir_r, 
                dir_u,
                dir_d,
                food_left,
                food_right,
                food_up,food_down]

        return np.array(state,dtype=float)


    def remember(self, state, action, reward, nxt_state, game_over):
        self.memory.append ((state, action, reward, nxt_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, nxt_states, game_overs = zip(*sample)
        self.train_short_memory( states, actions, rewards, nxt_states, game_overs)



    def train_short_memory(self, state, action, reward, nxt_state, game_over):

        self.trainer.train_step(state, action, reward, nxt_state, game_over)

    def get_action(self, state):
        self.eps = 100 - self.n_games

        move = [0,0,0]

        if random.randint(0,100) <  self.eps:
            idx = random.randint(0,2)
            move[idx] = 1 
        else:
            state =  torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            idx  = torch.argmax(prediction).item()

            move[idx] = 1

        return move



def train():
    scores = []
    record = 0
    mean_scores = []
    total_score = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        '''
            1- get cur state 
            2- call model.predict to get action
            3- call game.move(action)
            4- get new state
            5- train short mem
        
        '''

        cur_state = agent.get_state(game)

        move = agent.get_action(state=cur_state)

        reward, game_over, score = game.play_step(action=move)

        new_state = agent.get_state(game)

        agent.train_short_memory(cur_state,move, reward,new_state,game_over)

        agent.remember(cur_state,move, reward,new_state,game_over)

        if game_over:
            agent.n_games +=1 
            agent.train_long_memory()
            game.reset()

            if score > record:
                record = score
                agent.model.save()

                
            print(f'Game {agent.n_games}, Score is {score}, Highest score is: {record}')

            # TODO: PLOT score and mean score
            scores.append(score)
            total_score += score
            mean_scores.append( total_score / agent.n_games)
            plot(scores,mean_scores)



if __name__ == '__main__':
    train()
    


