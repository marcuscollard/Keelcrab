import gymnasium as gym

##################################
############ ENVS ################
##################################

# from .env_registry import MushrDriftRLEnvCfg, MushrDriftPlayEnvCfg
# from .visual import MushrVisualRLEnvCfg, MushrVisualPlayEnvCfg
# from .elevation import MushrElevationRLEnvCfg, MushrElevationPlayEnvCfg
# import wheeledlab_tasks.drifting.config.agents.mushr as mushr_drift_agents
# import wheeledlab_tasks.visual.config.agents.mushr as mushr_visual_agents
# import wheeledlab_tasks.elevation.config.agents.mushr as mushr_elevation_agents

from .env_registry import *


gym.register(
    id="Isaac-RL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MushrDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_drift_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrDriftPlayEnvCfg
    }
)