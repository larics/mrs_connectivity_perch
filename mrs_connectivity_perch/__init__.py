from gymnasium.envs.registration import register

register(
    id="Perch-v0",  # Name of the environment
    entry_point="mrs_connectivity_perch.envs:PerchV0",  # Correct path to the Homo class
    max_episode_steps=10000000000000000000000,
)

