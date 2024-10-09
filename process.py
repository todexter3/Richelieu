!pip install diplomacy
!pip install gymnasium
!pip install diplomacy
!pip install "stable-baselines3[extra]>=2.0.0a4"

import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from stable_baselines3.common.env_checker import check_env

import threadingu
import gymnasium as gym
import numpy as np


# import gym
from gymnasium import spaces
# gym import spaces

import subprocess
import os
import time
import signal
import atexit
import numpy as np

import grpc
import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format



class DiplomacyStrategyEnv(gym.Env):
    """
    The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    # Set these in ALL subclasses
    action_space = None
    # observation_space = None

    ### CUSTOM ATTRIBUTES

    def __init__(self):
        super(DiplomacyStrategyEnv, self).__init__()
        self.game = None
        self.open_game()
        # self.game=self.game2
        self.current_step = 0
        self.reward = 0
        # self.observation = None
        self._init_observation_space()
        self.player = 'FRANCE'
        self.max_units = len(self.game.get_state()['units'][self.player])
        self.units = self.game.get_state()['units'][self.player]
        self.max_locations = len(self.game.map.locs)
        self.max_actions = self.check_max_action()  # Adjust this according to the maximum possible actions
        # Define action space as a discrete space where each unit can select one action
        self.action_space = spaces.MultiDiscrete(self.max_actions)

        # self.reset()



    def open_game (self):
        self.game = Game(map='standard')

    def check_max_action(self):
        # l = self.game.get_state()['units'][self.player]
        possible_orders = self.game.get_all_possible_orders()
        add = []
        for power_name, power in self.game.powers.items():
        # # print(power_name ,"kdjf")
          if power_name ==self.player:
            alllocs = self.game.get_orderable_locations(power_name)
            if alllocs !=[]:
              for loc in alllocs:
                if loc in possible_orders:
                  act = possible_orders[loc]
                  if act !=[]:
                    add.append(len(act))
        if add==[]:
          add=[]
        return add

    def observation_data_to_observation(self):
        """
        """
        ### CONSTANTS
        NUMBER_OF_OPPONENTS = len(self.game.powers)
        NUMBER_OF_PROVINCES = len(self.game.map.locs)
        number_of_provinces = NUMBER_OF_PROVINCES

        observation = np.zeros(number_of_provinces * 3, dtype=int)

        for i, province in enumerate (self.game.map.locs):
            # simply for type hint and auto-completion
            # id - 1 because the ids begin at 1

            observation[i*3] = int(i)

            for num, power in enumerate (self.game.map.units):
              if province in self.game.get_state()['centers'][power]:
                observation[i * 3+1] = int(num)
            # the next is to check if that province is a supply center or not
              if province in self.game.map.scs:
                observation[i * 3 + 2] = 1


        reward = 0 #observation_data.previousActionReward
        done = self.game.is_game_done #observation_data.done
        info = {"Phase": self.game.get_current_phase()}
        # observations = np.array([2, 5, 9, 8, 0,9, 8,8, 8,0, 9,
        #                 0, 8,9, 0,9, 0, 9,9, 0])
        return observation, reward, done, info

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. action is in form of a list [3,5,2]i.e the choosing index of
        the selected action
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # truncated = self.current_step >= self.ep_length
        # return self.state, reward, terminated, truncated, {}
        prev_state = self.game.get_state()
        possible_orders= self.game.get_all_possible_orders()
        our_agent = self.player
        # unit_actions = np.split(action, self.num_units)
        # print(unit_actions)
        # Decode each unit's action
        actions_taken = []
        # self.units

        # Process each unit's action
        # for unit_action in unit_actions:
        #     command_type, target_location = self.decode_action(unit_action)
        #     # Apply the action logic for each unit
        #     # Example: update the game state based on the action
            # (This is where the game-specific logic would go)

        for power_name, power in self.game.powers.items():#zip( self.game.map.centers.keys(), self.game.map.units.values()):
            # print(power_name, centers)
            # possible_orders = self.game.get_all_possible_orders()
            chosen_action=[]
            if power_name != our_agent:
              power_orders = [
                  random.choice(possible_orders[loc])
                  for loc in self.game.get_orderable_locations(power_name)
                  if possible_orders[loc]
              ]
              # print(power_name, power_orders)
              self.game.set_orders(power_name, power_orders)
            else:
              alllocs=self.game.get_orderable_locations(power_name)
              for loc, actions in zip(alllocs,action):
                if loc in possible_orders:
                  act=possible_orders[loc]
                  # print(action,actions,len(act))
                  # print('______________',self.action_space)
                  if act !=[]:
                    if actions >= len(act):
                      actions = (len(act)) - 1
                    chosen_action.append(act[actions])


                  # Set the orders for each power
              self.game.set_orders(power_name, chosen_action)

        self.game.process()
        self.observation, _, _, info = self.observation_data_to_observation()
        current_state= self.game.get_state()
        # print(self.game.ordered_units)
        # print(prev_state)
        # print(current_state)
        self.calculate_reward(prev_state,actions_taken,current_state,our_agent)
        self.current_step += 1
        self.max_actions = self.check_max_action()
        self.action_space = spaces.MultiDiscrete(self.max_actions)
        # terminated = self.game.is_game_done
        truncated = False
        terminated = self.game.is_game_done

        return self.observation, self.reward, terminated, truncated, info

    def calculate_reward(self,previous_state, actions_taken, new_state,our_agent):

        check_new= new_state['influence'][our_agent]
        check_old = previous_state['influence'][our_agent]

        diff = list(set(check_new) - set(check_old))
        if diff !=[]:
          for i in diff:
            self.reward += 10 # it captures just a unit
            if i in self.game.map.scs:
              self.reward +=10 # it captures a supply center

        diff = list(set(check_old) - set(check_new))

        if diff !=[]:
          for i in diff:
            self.reward -=10

        for action in actions_taken:
          if len(action.split())>=3:
            if action.split()[2] == '-' and action[-1] in self.game.map.scs:
              self.reward +=5
            if action.split()[2] == 'S' and action[4] in self.game.map.scs:
              self.reward +=5

        return self.reward

    def reset(self, seed=None):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the space.
        """
        # super().reset(seed=seed)
        # super().reset(seed=seed)
        # if seed is not None:
        #     np.random.seed(seed)

        self.observation, _, _, _ = self.observation_data_to_observation()
        self.current_step
        self.max_actions = self.check_max_action()
        self.action_space = spaces.MultiDiscrete(self.max_actions)
        # self.open_game()

        return np.array(self.observation),{}


    def _init_observation_space(self):
        '''
        Observation space: [[province_id, owner, is_supply_center, has_unit] * number of provinces]
        The last 2 values represent the player id and the province to pick the order.
        Eg: If observation_space[2] is [5, 0, 0], then the second province belongs to player 5, is NOT a SC, and does NOT have a unit.
        '''
        observation_space_description = []
        NUMBER_OF_PLAYERS = len(self.game.powers)
        NUMBER_OF_PROVINCES =  len(self.game.map.locs)
        index_range = 82  # 0 to 81 inclusive
        value1_range = 7  # 0 to 6 inclusive
        value2_range = 2  # 0 or 1

        index_range = 82  # 0 to 81 inclusive
        value1_range = 7  # 0 to 6 inclusive
        value2_range = 2  # 0 or 1

        # Example dataset length
        data_length = 82 * 3  # Total number of elements in the dataset

        # Create the MultiDiscrete space
        self.observation_space = spaces.MultiDiscrete([index_range, value1_range, value2_range] * (data_length // 3))
        # print((self.observation_space))





def main():
  gm = DiplomacyStrategyEnv()
  # g=gm.reset()
  # print((g[0]))


if __name__ == "__main__":
    main()



env = DiplomacyStrategyEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

env = DiplomacyStrategyEnv()

obs, _ = env.reset()
# env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

GO_LEFT = list(np.random.randint(10, size=3))
# Hardcoded best agent: always go left!
n_steps = 20000
on=[0]
for step in range(n_steps):
# while not done:
    print(f"Step {step + 1}")
    act = env.action_space
    # print(act)
    GO_LEFT=[]
    if act !=[]:
      for y in act:
        y=int(y.sample())
        # print(y,act,'-------------------')
        GO_LEFT.append(y)
      Go_LEFT = list(GO_LEFT)

    # Hardcoded best agent: always go left!
    obs, reward, terminated, truncated, info = env.step(GO_LEFT)
    print("envifonsmnd obs ===",(on==obs).all())
    on=obs
    done = terminated
    print("obs=", obs, "reward=", reward, "done=", done)
    # env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(DiplomacyStrategyEnv, n_envs=1)

# model = PPO("MlpPolicy", vec_env, verbose=1,tensorboard_log='PPO' )
# model.learn(total_timesteps=25000,log_interval=1, tb_log_name='PPO',progress_bar = True)
# model.save("ppo_diplomacy")

# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_diplomacy")

obs = vec_env.reset()
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    # vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
