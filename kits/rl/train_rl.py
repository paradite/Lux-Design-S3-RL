import jax
import jax.numpy as jnp
import numpy as np
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams

def train_basic_env(num_episodes=2):  # Reduced episodes for faster testing
    """
    Train a basic RL agent using random actions.
    This is a minimal implementation to demonstrate the training loop structure.
    """
    # Initialize environment
    env = LuxAIS3Env()
    params = env.default_params
    print("Environment initialized with params:", params)
    
    # Initialize random key for JAX
    key = jax.random.PRNGKey(0)
    
    # Track basic metrics
    total_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)
        
        episode_reward = jnp.array(0.0)
        done = False
        step_count = 0
        
        while not done and step_count < params.max_steps_in_match:
            step_count += 1
            print(f"Episode {episode + 1}, Step {step_count}/{params.max_steps_in_match}")
            # Generate random keys for both players
            key, key_p0 = jax.random.split(key)
            key, key_p1 = jax.random.split(key)
            
            # For now, just generate random actions for both players
            # Shape: (max_units, 3) for each player
            # Generate actions as JAX arrays
            actions = {
                "player_0": jnp.array(jax.random.randint(key_p0, (params.max_units, 3), 0, 5), dtype=jnp.int32),
                "player_1": jnp.array(jax.random.randint(key_p1, (params.max_units, 3), 0, 5), dtype=jnp.int32)
            }
            
            # Step environment
            key, step_key = jax.random.split(key)
            # Step environment and handle rewards/done state
            step_result = env.step(step_key, state, actions, params)
            
            # Access components individually to avoid type issues
            next_obs = step_result[0]
            next_state = step_result[1]
            rewards = step_result[2]
            done_states = step_result[3]
            info = step_result[4]
            
            state = next_state
            # Print step info for debugging
            print(f"Step in episode {episode + 1}, current reward shape:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), rewards))
            
            # Handle rewards - keep as JAX array until end of episode
            if isinstance(rewards, dict):
                current_reward = rewards["player_0"]
            else:
                current_reward = rewards[0]
            episode_reward = episode_reward + current_reward
            
            # Handle done state using JAX operations
            if isinstance(done_states, dict):
                done = done_states["player_0"]
            else:
                done = done_states
            # Convert to bool only at the end
            done = bool(done)
            
            # done is already a boolean from step()
        
        # Convert episode reward to float at the end of episode
        final_reward = float(episode_reward)
        total_rewards.append(final_reward)
        print(f"Episode {episode + 1}/{num_episodes} complete, Reward: {final_reward}")
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
    
    # Save dummy model parameters (for demonstration)
    # In a real implementation, this would save learned policy parameters
    model_params = {
        "dummy_weights": np.zeros((1,)),
        "mean_reward": np.mean(total_rewards)
    }
    np.savez("kits/rl/model_params.npz", **model_params)
    print(f"Training complete. Mean reward: {np.mean(total_rewards)}")

if __name__ == "__main__":
    train_basic_env()
