import time

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


LIVES_DEFAULT = 8
MAX_DEFAULT = 30
TIME_DEFAULT = 20

class Trainer:
    """
        Used as an interface between the Generator and the Player to allow the first to train and use the latter
    """

    def __init__(self, env, timesteps, log_dir, model=None, preset=False):
        """

        :param env: Player Env Instance
        :param timesteps: Timesteps to train our Model
        """
        self.env = env
        self.timesteps = timesteps
        self.log_dir = log_dir
        self.model = model
        self.preset = preset
        self.iterations = 0

    def train(self, lr=None, er=None, default=False):
        """

        :param lr: Learning Rate For the PPO Model
        :param er: Entropy Loss Coef for the PPO Model
        :return: An Array with the average losses
        """
        if not default:
            train_env = self.env()
            if self.model is None:
                self.model = PPO('MultiInputPolicy', train_env, verbose=1, tensorboard_log=self.log_dir)
                self.model.learn(total_timesteps=self.timesteps)
                self.model.save(f'../../Player_models/V3.zip')
            else:
                print('Training Cont.')
                self.model = PPO.load(f'../../Player_models/V3.zip', env=train_env)
                self.model.learn(total_timesteps=self.timesteps)

            return [int(train_env.get_average_lives_lost()), int(train_env.get_average_time_lost()),
                    train_env.get_max_number_appeared()]
        else:
            return [LIVES_DEFAULT, TIME_DEFAULT, MAX_DEFAULT]

    def play(self, level, primes, max_number, default=False):
        if not default:
            self.preset = False
            eval_env = self.env(preset_level=[level, primes, max_number], training=False)
            state, _ = eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(state)
                state, reward, done, _, info = eval_env.step(action)

            return [eval_env.get_average_lives_lost(), eval_env.get_average_time_lost(), eval_env.board, eval_env.numbers_eaten]
        else:
            eval_env = self.env(preset_level=[level, primes, max_number], training=False)
            state, _ = eval_env.reset()
            done = False
            model = PPO.load(f'../../Player_models/V3.zip', env=eval_env)

            while not done:
                action, _ = model.predict(state)
                state, reward, done, _, info = eval_env.step(action)

            return [eval_env.get_average_lives_lost(), eval_env.get_average_time_lost(), eval_env.board,
                    eval_env.numbers_eaten]


    def save(self, path):
        self.model.save(path)
        return
