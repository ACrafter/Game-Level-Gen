import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from Environments.Generator.GenV3 import Generator
from Environments.Player.PlayerV3 import PlayerV3
from Environments.Player.PlayerTrainer import Trainer

# TODO: Fix The Prime Count Problem
# TODO: Fix the Diversity Problem
# TODO: Celebrate


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


PLY_LOGS = './player_logs/3.'
GEN_LOGS = './generator_logs/2.'
CHECKPOINT_DIR = './Generator_models/V3/'
GEN_STEPS = 5_000_000
PLY_STEPS = 500_000
GEN_FREEZE_AFTER = 1_000_000

Player = Trainer(PlayerV3, PLY_STEPS, PLY_LOGS)
env = Generator(Player, GEN_FREEZE_AFTER)
callback = TrainAndLoggingCallback(check_freq=GEN_FREEZE_AFTER + PLY_STEPS, save_path=CHECKPOINT_DIR)

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=GEN_LOGS, ent_coef=0.001)
model.learn(total_timesteps=GEN_STEPS, callback=callback)
