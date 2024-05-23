# If you are running this on a server you will not be able to use render_mode="human"
# instead use render_mode="rgb_array" since you cannot ouput to a display.

# WARNING: This may not run on M1 macs.
# If you really want to run the example, I recommend creating a seperate
# conda env and installing gym with pip install gym[atari, accept-rom-license] 
# But use the ex6 environment when working on the dqn file.

import gym
import time
import dqn
import torch
import torch.nn as nn



# stable_baselines3 have wrappers that simplifies 
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id, seed, idx, capture_video, run_name, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)  # Pass render_mode to gym.make()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

env_id = "ALE/Breakout-v5"
seed = 42
idx = 0
capture_video = False
run_name = "my_run"
render_mode = "human"  # Pass render_mode as an argument


env = gym.vector.SyncVectorEnv([make_env(env_id, seed, idx, capture_video, run_name, render_mode=render_mode)])




observation= env.reset(seed=42)
model_path =  "C:/Users/admin/Documents/GitHub/d7047e/lab4/runs/BreakoutNoFrameskip-v4__DQN_Breakout__1__1716313496/best_model_545562.pth"

state_dict = torch.load(model_path)
qn = dqn.QNetwork(env)
qn.load_state_dict(state_dict)
qn.eval()


observation = env.reset(seed=42)

for _ in range(1000):

    
    # Use the QNetwork to predict the action
    with torch.no_grad():
         q_values = qn(torch.tensor(observation, dtype=torch.float32)) # TODO: get q_values from the network you defined, what should the network receive as input?
         actions = torch.argmax(q_values, dim=1).cpu().numpy() # select actions with highest q value

    # Take a step in the environment using the chosen action
    observation, reward, done, info = env.step(actions)
   

    # Render the environment
    #env.render()

    if done:
        observation = env.reset(seed=42)

# Close the environment
env.close()
print("Executed environment successfully.")