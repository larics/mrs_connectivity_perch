import random
import sys
from os import path

import gymnasium as gym  # Ensure updated Gymnasium import
import numpy as np
import yaml

# This import has to be left for the script to find the registered environments of mrs_connectivity_perch
import mrs_connectivity_perch

save_dir = "/home/khawaja/HERO/CRC B1/perching t1/unperch/tests/perching_hero2_final"

def run(args, config_filename=None):
    # Load environment
    env_name = args.get('env')
    print(f"Initializing environment: {env_name}")

    # Pass the `args` configuration to the environment via kwargs
    env = gym.make(env_name, config=args)

    # Set random seeds for reproducibility
    seed = args.get('seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    # Environment reset
    obs, _ = env.reset()
    done = False
    t = 0

    # Fiedler value tracking for disconnection detection
    fiedler_threshold = 1e-3  # Threshold for "almost zero"
    consecutive_low_fiedler_count = 0
    required_consecutive_count = 5  # Number of consecutive low Fiedler values needed
    fiedler_check_interval = 5  # Check every N timesteps

    env.render()
    
    # Optional: Enable mission metrics (non-invasive)
    try:
        config_name = config_filename or "unknown_config"
        env.unwrapped.enable_mission_metrics(
            mission_name="perch_exploration_test",
            config_name=config_name,
            trial_number=1
        )
        print("📊 Mission metrics enabled")
    except Exception as e:
        print(f"📊 Mission metrics not available: {e}")

    while not done:
        if t > 30:
            env.unwrapped.controller()

            if t % 3 == 0:
                env.render()
                
            # Optional: Print metrics summary every 50 steps
            # if hasattr(env.unwrapped, 'mission_metrics') and env.unwrapped.mission_metrics and t % 50 == 0:
            #     env.unwrapped.mission_metrics.print_summary()

            # if t==200:
            #     env.unwrapped.swarm.add_danger_region(x_min=0.1, y_min=1.7, x_max=1.5, y_max=3.0, region_id="danger_zone_1")

        # print(t)
        t += 1

        # if t>850:
        #     print("Max timesteps reached, ending simulation.")
        #     break

    print("Simulation ended.")
    
    # Optional: Save mission metrics at the end
    # try:
    #     if hasattr(env.unwrapped, 'mission_metrics') and env.unwrapped.mission_metrics:
    #         env.unwrapped.mission_metrics.print_summary()
    #         metrics_file = env.unwrapped.save_mission_metrics()
    #         print(f" Final metrics saved to: {metrics_file}")
    # except Exception as e:
    #     print(f" Could not save metrics: {e}")


def main():
    """
    Entry point of the script. Reads configuration and runs the environment.
    """
    # Ensure the configuration file is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python test_script.py <config_file>")
        return

    # Load configuration file
    config_file = sys.argv[1]
    config_file_path = path.join(path.dirname(__file__), config_file)
    print(f"Loading configuration from: {config_file_path}")

    # Load the configuration
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Run the simulation using the default section of the config
    run(config, config_filename=config_file)


if __name__ == "__main__":
    main()
