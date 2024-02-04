import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Environments.Generator.GenV3 import Generator
from Environments.Player.PlayerV4 import Player
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


if __name__ == "__main__":
    PLY_LOGS = './player_logs/4.'
    GEN_LOGS = './final_generator_logs/V3.'
    CHECKPOINT_DIR = './V3.'
    GEN_STEPS = 500_000
    PLY_STEPS = 1
    GEN_FREEZE_AFTER = 1_000_000

    Player = Trainer(Player, PLY_STEPS, PLY_LOGS)


    def create_gen_env():
        return Generator(True, Player, GEN_FREEZE_AFTER)


    env = SubprocVecEnv([lambda: create_gen_env() for _ in range(5)])
    # env = Generator(True, Player, GEN_FREEZE_AFTER)
    callback = TrainAndLoggingCallback(check_freq=500_000, save_path=CHECKPOINT_DIR)

    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=GEN_LOGS)
    model.learn(total_timesteps=GEN_STEPS, callback=callback)
    model.save('./Final_Generator_models/V3/Max 55/.')
