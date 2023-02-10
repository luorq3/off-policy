import numpy as np
import gym
from gym import spaces
import pygame
from offpolicy.envs.combatV2.entities import Ship, Fort
from offpolicy.envs.combatV2.rendering import Viewer
import time


"""
9*9的地图，左边4*9为进攻方，右边4*9为防守方，正中间1*9不可跨越
"""
class Game(gym.Env):
    def __init__(self):
        super(Game, self).__init__()
        self.map_size = [9, 9]
        self.n_ships = 3
        self.n_forts = 3

        self.obs_dim = 16
        self.shared_obs_dim = 16
        self.action_space = [spaces.Discrete(8) for _ in range(self.n_ships)]
        self.observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]
        self.share_observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.shared_obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]

        self._init()

    def _init(self):
        self.ships = []
        self.forts = []
        self.map = np.zeros(self.map_size, dtype=int) - 1
        self._agent_init()
        self.agents = self.ships + self.forts
        for agent in self.agents:
            self.map[agent.x, agent.y] = agent.idx

        # fort action
        self._random_fort_action()

        self._viewer = Viewer()

    def _random_fort_action(self):
        fort_actions = np.random.randint(0, 36, size=self.n_forts)
        for fort, action in zip(self.forts, fort_actions):
            fort.action = action

    def _agent_init(self):
        ship_xs = np.random.choice(range(9), self.n_ships, replace=False)
        ship_ys = np.random.choice(range(4), self.n_ships)
        for idx, (x, y) in enumerate(zip(ship_xs, ship_ys)):
            self.ships.append(Ship(idx, x, y))

        fort_xs = np.random.choice(range(9), self.n_forts, replace=False)
        fort_ys = np.random.choice(range(5, 9), self.n_forts)
        for idx, (x, y) in enumerate(zip(fort_xs, fort_ys)):
            self.forts.append(Fort(self.n_ships+idx, x, y))

    def _get_state(self):
        """
        states: [自己的坐标，队友的坐标，对手的生命值，被攻击的位置，自己的索引值]
        """
        states = []
        for ship in self.ships:
            if ship.is_dead:
                return np.zeros((3, self.obs_dim)) - 1
            pos = [[ship.x, ship.y]]
            for i_ship in self.ships:
                if i_ship.idx != ship.idx:
                    if i_ship.is_dead:
                        pos.append([-1, -1])
                    else:
                        pos.append([i_ship.x, i_ship.y])
            forts_hp = []
            for i_fort in self.forts:
                if not i_fort.is_dead:
                    forts_hp.append(i_fort.hp)
                else:
                    forts_hp.append(0)
            fired_pos = []
            for i_fort in self.forts:
                action = i_fort.action
                if not i_fort.can_fire or i_fort.is_dead:
                    fired_pos.append([-1, -1])
                else:
                    fired_pos.append([action % 9, action // 9])
            pos = np.array(pos, dtype=np.float).flatten()
            forts_hp = np.array(forts_hp).flatten()
            fired_pos = np.array(fired_pos, dtype=np.float).flatten()
            state = np.concatenate((pos, forts_hp, fired_pos, [ship.idx]))
            states.append(state)

        states = np.array(states)
        return states

    def step(self, actions, render=False):
        print(f"step function: render={render}")
        reward = 0
        individual_rewards = [0 for _ in range(self.n_ships)]
        infos = {i: {} for i in range(self.n_ships)}

        fort_actions = []
        for fort in self.forts:
            fort_actions.append(fort.action)
        render_actions = np.concatenate((actions, fort_actions))
        if render:
            self.render(actions=render_actions)

        for i, action in enumerate(actions):
            agent = self.ships[i]

            if agent.is_dead:
                continue

            # 0-noop, 1-up, 2-down, 3-left, 4-right, 5~7-every fort
            if action >= 5:
                action -= 5
                fired_agent = self.agents[action + 3]
                if fired_agent.is_dead:
                    continue
                fired_agent.hp -= agent.dmg
                reward += agent.dmg
                individual_rewards[i] += agent.dmg
                if fired_agent.is_dead:
                    self.map[fired_agent.x, fired_agent.y] = -1
                    reward += 10
                    individual_rewards[i] += 10
            elif action == 1:
                # 往上移动一步
                if agent.x > 0 and self.map[agent.x - 1, agent.y] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.x -= 1
                    self.map[agent.x, agent.y] = agent.idx
                # 碰撞reward-1
                # else:
                #     reward -= 1
                #     individual_rewards[i] -= 1
            elif action == 2:
                # 往下移动一步
                if agent.x < self.map_size[0] - 1 and self.map[agent.x + 1, agent.y] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.x += 1
                    self.map[agent.x, agent.y] = agent.idx
                # 碰撞reward-1
                # else:
                #     reward -= 1
                #     individual_rewards[i] -= 1
            elif action == 3:
                # 往左移动一步
                if agent.y > 0 and self.map[agent.x, agent.y - 1] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.y -= 1
                    self.map[agent.x, agent.y] = agent.idx
                # else:
                #     reward -= 1
                #     individual_rewards[i] -= 1
            elif action == 4:
                # 往右移动一步
                if agent.y < self.map_size[1] // 2 - 1 and self.map[agent.x, agent.y + 1] == -1:
                    self.map[agent.x, agent.y] = -1
                    agent.y += 1
                    self.map[agent.x, agent.y] = agent.idx
                # else:
                #     reward -= 1
                #     individual_rewards[i] -= 1

        for fort in self.forts:
            if fort.is_dead:
                continue

            fired_x = fort.action % 9
            fired_y = fort.action // 9
            if self.map[fired_x, fired_y] >= 0 and not self.ships[self.map[fired_x, fired_y]].is_dead:
                fired_agent = self.ships[self.map[fired_x, fired_y]]
                fired_agent.hp -= fort.dmg
                reward -= fort.dmg
                individual_rewards[fired_agent.idx] -= fort.dmg
                if fired_agent.is_dead:
                    self.map[fired_x, fired_y] = -1
                    reward -= 10
                    individual_rewards[fired_agent.idx] -= 10

        done = True
        if np.all([fort.is_dead for fort in self.forts]):
            reward += 100
        elif np.all([ship.is_dead for ship in self.ships]):
            reward -= 100
        else:
            done = False

        for i, info in infos.items():
            info['individual_reward'] = individual_rewards[i]

        # update fort action
        self._random_fort_action()
        obs = self._get_state()
        rewards = np.array([[reward] for _ in range(self.n_ships)], dtype=np.float32)
        dones = (np.array([[ship.is_dead] for ship in self.ships]) | done).astype(np.float32)
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
