import os
import numpy as np
import gym
from gym import spaces
import pygame
from offpolicy.envs.combatV3.entities import Ship, Fort
from offpolicy.envs.combatV3.rendering import Viewer
import time
import yaml


conf_path = os.path.dirname(__file__) + '/config.yaml'

def load_conf(path=conf_path):
    with open(path) as s:
        conf = yaml.safe_load(s)
    return conf


class Game(gym.Env):
    def __init__(self):
        super(Game, self).__init__()
        conf = load_conf()
        self.map_size = conf['map_size']
        self.n_ships = conf['n_ships']
        self.n_forts = conf['n_forts']

        self.obs_dim = self._get_obs_dim()
        self.shared_obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.n_ships)]
        self.observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]
        self.share_observation_space = [spaces.Box(low=np.inf, high=np.inf, shape=(self.shared_obs_dim,), dtype=np.float32) for _ in range(self.n_ships)]

    def _get_obs_dim(self):
        return (self.n_ships + self.n_forts) * 3

    def _get_action_dim(self):
        return self.n_forts + 5

    def _init(self):
        self.ships = []
        self.forts = []
        self.ship_map = np.zeros(self.map_size, dtype=int) - 1
        self.fort_map = np.zeros(self.map_size, dtype=int) - 1
        self._agent_init()
        for ship in self.ships:
            self.ship_map[ship.x, ship.y] = ship.idx
        for fort in self.forts:
            self.fort_map[fort.x, fort.y] = fort.idx

        # fort action
        self._random_fort_action()

        self._viewer = Viewer(self.map_size)

    def _random_fort_action(self):
        fort_actions = np.random.randint(0, self.map_size[0] * self.map_size[1], size=self.n_forts)
        for fort, action in zip(self.forts, fort_actions):
            fort.action = action

    def _agent_init(self):
        ship_xs = np.random.choice(range(self.map_size[0]), self.n_ships, replace=False)
        ship_ys = np.random.choice(range(self.map_size[1]), self.n_ships)
        for idx, (x, y) in enumerate(zip(ship_xs, ship_ys)):
            self.ships.append(Ship(idx, x, y))

        fort_xs = np.random.choice(range(self.map_size[0]), self.n_forts, replace=False)
        fort_ys = np.random.choice(range(self.map_size[1]), self.n_forts)
        for idx, (x, y) in enumerate(zip(fort_xs, fort_ys)):
            self.forts.append(Fort(idx, x, y))

    def _get_state(self):
        """
        states: [自己的坐标，队友的坐标，被攻击的位置，对手的生命值，自己的索引值one-hot]
        """
        states = []
        for ship in self.ships:
            if ship.is_dead:
                return np.zeros((self.n_ships, self.obs_dim)) - 1
            pos = [[ship.x, ship.y]]
            for i_ship in self.ships:
                if i_ship.idx != ship.idx:
                    if i_ship.is_dead:
                        pos.append([-1, -1])
                    else:
                        pos.append([i_ship.x, i_ship.y])
            fired_pos = []
            for i_fort in self.forts:
                action = i_fort.action
                if not i_fort.can_fire or i_fort.is_dead:
                    fired_pos.append([-1, -1])
                else:
                    fired_pos.append([action % self.map_size[0], action // self.map_size[0]])
            forts_hp = []
            for i_fort in self.forts:
                if not i_fort.is_dead:
                    forts_hp.append(i_fort.hp)
                else:
                    forts_hp.append(0)
            pos = np.array(pos, dtype=np.float).flatten()
            fired_pos = np.array(fired_pos, dtype=np.float).flatten()
            forts_hp = np.array(forts_hp).flatten()
            one_hot_idx = np.zeros(self.n_ships)
            one_hot_idx[ship.idx] = 1
            state = np.concatenate((pos, fired_pos, forts_hp, one_hot_idx))
            states.append(state)

        states = np.array(states)
        return states

    def step(self, actions, render=False):
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
            ship = self.ships[i]

            if ship.is_dead:
                continue

            # 0-noop, 1-up, 2-down, 3-left, 4-right, 5~inf-every fort
            if action >= 5:
                action -= 5
                fired_fort = self.forts[action]
                if fired_fort.is_dead:
                    reward -= 1
                    individual_rewards[i] -= 1
                    continue
                real_dmg = ship.dmg if fired_fort.hp > ship.dmg else fired_fort.hp
                fired_fort.hp -= real_dmg
                reward += real_dmg
                individual_rewards[i] += real_dmg
                if fired_fort.is_dead:
                    self.fort_map[fired_fort.x, fired_fort.y] = -1
                    reward += 10
                    individual_rewards[i] += 10
            elif action == 1:
                # 往上移动一步
                if ship.x > 0 and self.ship_map[ship.x - 1, ship.y] == -1:
                    self.ship_map[ship.x, ship.y] = -1
                    ship.x -= 1
                    self.ship_map[ship.x, ship.y] = ship.idx
                # 碰撞reward-1
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 2:
                # 往下移动一步
                if ship.x < self.map_size[0] - 1 and self.ship_map[ship.x + 1, ship.y] == -1:
                    self.ship_map[ship.x, ship.y] = -1
                    ship.x += 1
                    self.ship_map[ship.x, ship.y] = ship.idx
                # 碰撞reward-1
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 3:
                # 往左移动一步
                if ship.y > 0 and self.ship_map[ship.x, ship.y - 1] == -1:
                    self.ship_map[ship.x, ship.y] = -1
                    ship.y -= 1
                    self.ship_map[ship.x, ship.y] = ship.idx
                else:
                    reward -= 1
                    individual_rewards[i] -= 1
            elif action == 4:
                # 往右移动一步
                if ship.y < self.map_size[1] - 1 and self.ship_map[ship.x, ship.y + 1] == -1:
                    self.ship_map[ship.x, ship.y] = -1
                    ship.y += 1
                    self.ship_map[ship.x, ship.y] = ship.idx
                else:
                    reward -= 1
                    individual_rewards[i] -= 1

        for fort in self.forts:
            if fort.is_dead or not fort.can_fire:
                continue

            fired_x = fort.action % self.map_size[0]
            fired_y = fort.action // self.map_size[0]
            if self.ship_map[fired_x, fired_y] >= 0 and not self.ships[self.ship_map[fired_x, fired_y]].is_dead:
                fired_ship = self.ships[self.ship_map[fired_x, fired_y]]
                real_dmg = fort.dmg if fired_ship.hp > fort.dmg else fired_ship.hp
                fired_ship.hp -= real_dmg
                reward -= real_dmg
                individual_rewards[fired_ship.idx] -= fort.dmg
                if fired_ship.is_dead:
                    self.ship_map[fired_x, fired_y] = -1
                    reward -= 10
                    individual_rewards[fired_ship.idx] -= 10

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
        self._viewer.draw_surface(self.ships + self.forts, actions)
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
