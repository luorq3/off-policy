import random

import numpy as np
import gym
from gym import spaces
import pygame
from offpolicy.envs.combat.entities import Ship, Fort
from offpolicy.envs.combat.rendering import Viewer
import time


"""
9*9的地图，左边4*9为进攻方，右边4*9为防守方，正中间1*9不可跨越
"""
class Game(gym.Env):
    def __init__(self):
        super(Game, self).__init__()
        self.map_size = [9, 9]
        self.n_ships = 3
        self.n_forts = 2

        self.obs_dim = 20
        self.shared_obs_dim = 20

        self._init()
        self.action_space = [spaces.Discrete(40) for _ in range(self.n_ships)]
        self.observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]
        self.share_observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.shared_obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]

    def _init(self):
        self.ships = []
        self.forts = []
        self.map = np.zeros(self.map_size, dtype=int) - 1
        self._agent_init()
        self.agents = self.ships + self.forts
        for i, agent in enumerate(self.agents):
            agent.idx = i
            self.map[agent.x, agent.y] = agent.idx
        self.death_mask = [False] * len(self.agents)

        # fort action
        self.fort_actions = np.random.randint(0, 36, size=self.n_forts)

        self._viewer = Viewer()

    def _agent_init(self):
        ship_xs = np.random.choice(range(9), self.n_ships, replace=False)
        ship_ys = np.random.choice(range(4), self.n_ships)
        for x, y in zip(ship_xs, ship_ys):
            self.ships.append(Ship(x, y))

        fort_xs = np.random.choice(range(9), self.n_forts, replace=False)
        fort_ys = np.random.choice(range(5, 9), self.n_forts)
        for x, y in zip(fort_xs, fort_ys):
            self.forts.append(Fort(x, y))

    def _adjacent_pos(self, x, y):
        pos_set = []
        x_set = [x - 1, x, x + 1]
        y_set = [y - 1, y, y + 1]
        for xe in x_set:
            if xe < 0 or xe >= self.map_size[0]:
                continue
            for ye in y_set:
                if ye < 0 or ye >= self.map_size[1]:
                    continue
                pos_set.append([xe, ye])
        return pos_set

    def _get_state(self):
        """
        states: 目前只是单纯将每个agent的坐标及剩余生命值拼接起来。
        每个agent都观测到全局states，并且都一样。
        """
        states = []
        for agent in self.agents:
            if agent.is_dead:
                state = [0, 0, 0]
            else:
                state = [agent.x, agent.y, agent.hp]
            states.append(state)
        fired_pos = []
        for action in self.fort_actions:
            fired_pos.append([action % 9, action // 9])
        # 将所有左边和生命值拼接并打平
        states = np.array(states, dtype=np.float).flatten()
        fired_pos = np.array(fired_pos, dtype=np.float).flatten()
        states = np.concatenate((states, fired_pos))
        # 重复
        full_states = []
        for i in range(self.n_ships):
            full_states.append(np.append(states, i))

        # states = np.expand_dims(states, 0).repeat(self.n_ships, axis=0)
        full_states = np.array(full_states)
        return full_states

    def step(self, actions, render=False):
        reward = 0
        individual_rewards = [0 for _ in range(self.n_ships)]
        infos = {i: {} for i in range(self.n_ships)}

        render_actions = np.concatenate((actions, self.fort_actions))
        if render:
            self.render(actions=render_actions)

        for i, action in enumerate(actions):
            agent = self.ships[i]

            if self.death_mask[i]:
                continue

            # 0-35为攻击，输出地图上的一个位置
            if action < 36:
                fired_x = action // 9
                fired_y = action % 9 + 5
                # 对于被攻击的点，产生全额伤害，周围的8个点产生一半的伤害值
                pos_set = self._adjacent_pos(fired_x, fired_y)
                for x, y in pos_set:
                    if self.map[x, y] >= 0 and not self.death_mask[self.map[x, y]]:
                        fired_agent = self.agents[self.map[x, y]]
                        if x == fired_x and y == fired_y:
                            fired_agent.hp -= agent.dmg
                            reward += agent.dmg
                            individual_rewards[i] += agent.dmg
                        else:
                            fired_agent.hp -= (agent.dmg // 2)
                            reward += (agent.dmg // 2)
                            individual_rewards[i] += (agent.dmg // 2)
                        if fired_agent.hp <= 0:
                            self.death_mask[fired_agent.idx] = True
                            self.map[x, y] = -1
                            reward += 10
                            individual_rewards[i] += 10
            elif action == 36:
                # 往上移动一步
                if agent.x > 0 and self.map[agent.x - 1, agent.y] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.x -= 1
                    self.map[agent.x, agent.y] = agent.idx
                # 碰撞reward-1
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 37:
                # 往下移动一步
                if agent.x < self.map_size[0] - 1 and self.map[agent.x + 1, agent.y] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.x += 1
                    self.map[agent.x, agent.y] = agent.idx
                # 碰撞reward-1
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 38:
                # 往左移动一步
                if agent.y > 0 and self.map[agent.x, agent.y - 1] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.y -= 1
                    self.map[agent.x, agent.y] = agent.idx
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 39:
                # 往右移动一步
                if agent.y < self.map_size[1] // 2 - 1 and self.map[agent.x, agent.y + 1] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.y += 1
                    self.map[agent.x, agent.y] = agent.idx
                else:
                    reward -= 1
                    individual_rewards[i] -= 1

        for fort, action in zip(self.forts, self.fort_actions):
            if self.death_mask[fort.idx]:
                continue

            fired_x = action % 9
            fired_y = action // 9
            if self.map[fired_x, fired_y] >= 0 and not self.death_mask[self.map[fired_x, fired_y]]:
                fired_agent = self.agents[self.map[fired_x, fired_y]]
                fired_agent.hp -= fort.dmg
                reward -= fort.dmg
                individual_rewards[fired_agent.idx] -= fort.dmg
                if fired_agent.hp <= 0:
                    self.death_mask[fired_agent.idx] = True
                    self.map[fired_x, fired_y] = -1
                    reward -= 10
                    individual_rewards[fired_agent.idx] -= 10

        done = True
        if np.all(self.death_mask[self.n_ships:]):
            reward += 100
        elif np.all(self.death_mask[:self.n_ships]):
            reward -= 100
        else:
            done = False

        for i, info in infos.items():
            info['individual_reward'] = individual_rewards[i]

        # update fort action
        self.fort_actions = np.random.randint(0, 36, size=self.n_forts)
        obs = self._get_state()
        rewards = np.array([[reward] for _ in range(self.n_ships)], dtype=np.float32)
        dones = (np.array(self.death_mask[:self.n_ships]) | done).astype(np.float32)
        return obs, rewards, dones, infos

    def reset(self):
        self._init()
        return self._get_state()

    def render(self, mode='human', actions=None):
        time.sleep(1)
        self._viewer.draw_surface(self.agents, actions)
        if self._viewer.display is None:
            self._viewer.make_display()
        self._viewer.update_display()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def close(self):
        if self._viewer is not None and self._viewer.display is not None:
            pygame.display.quit()
