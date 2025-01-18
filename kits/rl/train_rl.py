import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from typing import Dict, Any, Tuple, List
from luxai_s3.state import EnvObs
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState, EnvObs
from policy import create_policy, sample_action, update_step

def train_basic_env(num_episodes: int = 100) -> None:
    """
    Train a policy gradient agent for the Lux AI Season 3 environment.
    Uses a simple policy network to learn unit movement strategies.
    
    Args:
        num_episodes: Number of episodes to train for
        
    The agent learns through direct policy gradient updates, using:
    - Policy network for action selection
    - Episode-based rewards for policy updates
    - Experience buffer for stable learning
    - Reward normalization for training stability
    
    Returns:
        None. Saves trained model parameters to kits/rl/model_params.npz
    
    Note:
        Throughout this function, obs is of type Dict[str, EnvObs] where:
        - The keys are "player_0" and "player_1"
        - The values are EnvObs objects containing the observation for each player
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize environment and types
    env: LuxAIS3Env = LuxAIS3Env()
    params: EnvParams = env.default_params
    
    # Type annotation for obs that will be used throughout the function
    # Note: obs is Dict[str, EnvObs] where keys are "player_0" and "player_1"
    obs: Dict[str, EnvObs]
    
    logging.info("Starting training with parameters:")
    logging.info(f"Max steps per match: {params.max_steps_in_match}")
    logging.info(f"Max units: {params.max_units}")
    logging.info(f"Map size: {params.map_width}x{params.map_height}")
    logging.info(f"Number of teams: {params.num_teams}")
    
    # Initialize random key for JAX
    key: chex.PRNGKey = jax.random.PRNGKey(0)
    
    # Initialize policy network and optimizer
    key, policy_key = jax.random.split(key)
    policy, policy_state, optimizer = create_policy(policy_key)
    
    # Track metrics
    total_rewards: List[float] = []
    episode_losses: List[float] = []
    
    # Initialize experience buffers for the episode
    episode_observations: List[Dict[str, EnvObs]] = []
    episode_actions: List[chex.Array] = []
    episode_rewards: List[float] = []
    
    # Buffer for collecting experience across episodes
    buffer_size = 1000
    all_observations: List[Dict[str, EnvObs]] = []
    all_actions: List[chex.Array] = []
    all_rewards: List[float] = []
    
    for episode in range(num_episodes):
        # Reset environment - returns Dict[str, EnvObs] and EnvState
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)  # obs is already Dict[str, EnvObs]
        
        episode_reward = jnp.array(0.0)
        done = False
        step_count = 0
        
        while not done and step_count < params.max_steps_in_match:
            step_count += 1
            # Generate keys for action sampling
            key, key_p0, key_p1 = jax.random.split(key, 3)
            
            # Get observation for player_0 and sample actions
            # obs is Dict[str, EnvObs], so we can directly index it
            p0_obs = {"player_0": obs["player_0"]}  # Create single-player observation dict
            p0_actions = sample_action(policy, policy_state.params, p0_obs, key_p0)
            
            # Convert to full action format (movement + sap direction)
            p0_full_actions = jnp.zeros((params.max_units, 3), dtype=jnp.int32)
            p0_full_actions = p0_full_actions.at[:, 0].set(p0_actions)
            
            # Random opponent actions
            actions = {
                "player_0": p0_full_actions,
                "player_1": jnp.array(jax.random.randint(key_p1, (params.max_units, 3), 0, 5), dtype=jnp.int32)
            }
            
            # Store experience
            episode_observations.append(obs)  # Store full observation dictionary
            episode_actions.append(p0_actions)
            
            # Step environment
            key, step_key = jax.random.split(key)
            # Step environment - returns (obs, state, reward, done_dict, info)
            step_result = env.step(step_key, state, actions, params)
            
            # Unpack step result components
            next_obs_dict, next_state, rewards, done_dict, info = step_result
            
            # Update state and observation
            state = next_state
            obs = next_obs_dict  # next_obs_dict is already Dict[str, EnvObs]
            
            # Get reward and update episode reward (rewards is Dict[str, float])
            current_reward = rewards["player_0"]
            episode_reward = episode_reward + current_reward
            
            # Check termination status
            done = done_dict["player_0"]["terminated"] or done_dict["player_0"]["truncated"]
            
            # Reward handling is done above
            
            # done is handled above using terminated/truncated dicts
        
        # Convert episode reward to float and store
        final_reward = float(episode_reward)
        total_rewards.append(final_reward)
        
        # Assign rewards to all steps in episode
        step_rewards = [final_reward] * len(episode_observations)
        
        # Add episode data to overall buffers
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        all_rewards.extend(step_rewards)
        
        # Clear episode buffers
        episode_observations = []
        episode_actions = []
        
        # Update policy if we have enough experience
        if len(all_observations) >= buffer_size:
            # Convert to arrays and normalize rewards
            obs_array = jnp.array(all_observations)
            action_array = jnp.array(all_actions)
            reward_array = jnp.array(all_rewards)
            reward_array = (reward_array - reward_array.mean()) / (reward_array.std() + 1e-8)
            
            # Update policy
            policy_state, loss = update_step(
                policy, policy_state,
                obs_array, action_array, reward_array,
                optimizer
            )
            episode_losses.append(float(loss))
            
            # Clear buffers
            all_observations = []
            all_actions = []
            all_rewards = []
        
        # Log progress every 10 episodes
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(total_rewards[-10:])
            logging.info("-" * 40)
            logging.info(f"Episode {episode + 1}/{num_episodes}")
            logging.info(f"Mean reward (last 10): {mean_reward:.2f}")
            logging.info(f"Latest episode reward: {final_reward:.2f}")
            if episode_losses:
                logging.info(f"Latest loss: {episode_losses[-1]:.4f}")
                logging.info(f"Total policy updates: {len(episode_losses)}")
            
            # Log detailed metrics
            logging.info("Episode Statistics:")
            logging.info(f"- Total steps: {step_count}")
            logging.info(f"- Active units: {len(episode_observations)}")
            logging.info(f"- Buffer size: {len(all_observations)}/{buffer_size}")
            logging.info("-" * 40)
    
    # Save trained policy parameters
    # Note: Using "dummy_weights" key to match what main.py expects
    model_params = {
        "dummy_weights": jax.device_get(policy_state.params),  # Save policy params as dummy_weights for compatibility
        "mean_reward": np.mean(total_rewards),
        "hidden_dims": policy.hidden_dims,  # Save network architecture for inference
        "max_units": params.max_units,  # Save environment config for inference
        "num_actions": 5  # Save action space size for inference
    }
    np.savez("kits/rl/model_params.npz", **model_params)
    print(f"Training complete. Mean reward: {np.mean(total_rewards)}")
    print(f"Saved model parameters to kits/rl/model_params.npz")

if __name__ == "__main__":
    train_basic_env()
