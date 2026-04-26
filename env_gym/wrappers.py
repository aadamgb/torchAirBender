from stable_baselines3.common.vec_env import VecEnv
from env_gym.trajectory_tracking import TrajTrckEnv

class TorchVecEnv(VecEnv):
    """
    Thin shim that makes TrajTrckEnv look like an SB3 VecEnv.
    The actual parallelism happens on GPU inside TrajTrckEnv.
    """
    def __init__(self, env: TrajTrckEnv):
        self.metadata = {"render_modes": ["human"]}
        self.env = env
        super().__init__(
            num_envs          = env.num_envs,
            observation_space = env.observation_space,
            action_space      = env.action_space,
        )
        self.render_mode       = env.render_mode

    def get_attr(self, attr_name, indices=None):
        val = getattr(self.env, attr_name)
        return [val] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def reset(self):
        obs, _ = self.env.reset()
        return obs   

    def step_async(self, actions):
        self._actions = actions

    # def step_wait(self):
    #     obs, reward, terminated, truncated, info = self.env.step(self._actions)
    #     done = terminated | truncated

    #     infos = []
    #     for i in range(self.num_envs):
    #         d = {
    #             "pos_err": info[i]["pos_err"],
    #             "vel_err": info[i]["vel_err"],
    #         }
    #         if terminated[i] or truncated[i]:
    #             d["terminal_observation"] = obs[i]
    #             # SB3 logs anything inside "episode" key automatically
    #             d["episode"] = {
    #                 "r": reward[i],           # episode reward
    #                 "l": self.env.t,          # episode length
    #                 "pos_err": info[i]["pos_err"],
    #                 "vel_err": info[i]["vel_err"],
    #             }
    #         infos.append(d)

    #     return obs, reward, done, infos
    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)
        done = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        return obs, reward, done, infos
    
    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return [None] * self.num_envs
    
    def render(self, mode="human"):
        return self.env.render()