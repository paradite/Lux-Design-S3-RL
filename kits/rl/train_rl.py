import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
import os
import flax
from typing import Dict, Any, Tuple, List, Union
from luxai_s3.state import EnvObs, EnvState
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from policy import create_policy, sample_action, update_step
import logging

def train_basic_env(num_episodes: int = 20) -> None:
    """Train a policy gradient agent for the Lux AI Season 3 environment."""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize environment and policy
    env: LuxAIS3Env = LuxAIS3Env()
    params: EnvParams = env.default_params
    
    # Initialize random key for JAX
    rng = jax.random.PRNGKey(0)
    rng, policy_rng = jax.random.split(rng)
    policy, policy_state, optimizer = create_policy(policy_rng)
    
    # Initialize metrics and buffers
    total_rewards: List[float] = []
    episode_losses: List[float] = []
    buffer_size = 1000
    
    # Initialize experience buffers
    episode_observations: List[Dict[str, Dict[str, Any]]] = []
    episode_actions: List[jnp.ndarray] = []
    episode_rewards: List[float] = []
    
    all_observations: List[Dict[str, Dict[str, Any]]] = []
    all_actions: List[jnp.ndarray] = []
    all_rewards: List[float] = []
    
    # Log training parameters
    logging.info("Starting training with parameters:")
    logging.info(f"Max steps per match: {params.max_steps_in_match}")
    logging.info(f"Max units: {params.max_units}")
    logging.info(f"Map size: {params.map_width}x{params.map_height}")
    logging.info(f"Number of teams: {params.num_teams}")
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        raw_obs, state = env.reset(reset_rng, params)
        
        # Get observation dictionary from environment
        obs_dict = env.get_obs(state, params)
        
        # Convert observation to policy format
        obs = convert_obs_to_dict(obs_dict)
        
        # Alternate between training as player_0 and player_1
        current_player = "player_0" if episode % 2 == 0 else "player_1"
        opponent_player = "player_1" if current_player == "player_0" else "player_0"
        current_team_idx = 0 if current_player == "player_0" else 1
        opponent_team_idx = 1 - current_team_idx
        
        logging.info(f"Episode {episode} - Training as {current_player}")
        
        # Log observation structure for debugging
        logging.info(f"Episode {episode} observation structure:")
        for player in [current_player, opponent_player]:
            player_obs = obs[player]
            logging.info(f"{player} observation shapes:")
            logging.info(f"  Units position: {player_obs['units']['position'].shape}")
            logging.info(f"  Units energy: {player_obs['units']['energy'].shape}")
            logging.info(f"  Units mask: {player_obs['units_mask'].shape}")
            logging.info(f"  Map features energy: {player_obs['map_features']['energy'].shape}")
            logging.info(f"  Map features tile type: {player_obs['map_features']['tile_type'].shape}")
            logging.info(f"  Sensor mask: {player_obs['sensor_mask'].shape}")
            logging.info(f"  Relic nodes: {player_obs['relic_nodes'].shape}")
            logging.info(f"  Relic nodes mask: {player_obs['relic_nodes_mask'].shape}")
        
        episode_reward = 0.0
        done = False
        step_count = 0
        
        # Episode loop
        while not done and step_count < params.max_steps_in_match:
            step_count += 1
            
            # Generate keys for action sampling
            rng, action_rng, opponent_rng = jax.random.split(rng, 3)
            
            # Sample actions for current player using the policy network
            current_actions = sample_action(policy, policy_state, obs, current_player, action_rng)
            
            # Convert to full action format (movement + sap direction)
            # Remove batch dimension since we're processing one step at a time
            current_actions_unbatched = current_actions[0]  # Shape: (max_units,)
            current_full_actions = jnp.zeros((params.max_units, 3), dtype=jnp.int32)
            current_full_actions = current_full_actions.at[:, 0].set(current_actions_unbatched)
            
            # Random opponent actions
            opponent_full_actions = jnp.zeros((params.max_units, 3), dtype=jnp.int32)
            opponent_full_actions = opponent_full_actions.at[:, 0].set(
                jax.random.randint(opponent_rng, (params.max_units,), 0, 5)
            )
            
            # Create action dictionary for both players
            actions = {
                current_player: np.array(current_full_actions),
                opponent_player: np.array(opponent_full_actions)
            }
            
            # Store experience
            episode_observations.append({
                "player_0": {
                    "units": {
                        "position": jnp.array(obs["player_0"]["units"]["position"]),
                        "energy": jnp.array(obs["player_0"]["units"]["energy"])
                    },
                    "units_mask": jnp.array(obs["player_0"]["units_mask"]),
                    "map_features": {
                        "energy": jnp.array(obs["player_0"]["map_features"]["energy"]),
                        "tile_type": jnp.array(obs["player_0"]["map_features"]["tile_type"])
                    },
                    "sensor_mask": jnp.array(obs["player_0"]["sensor_mask"]),
                    "team_points": jnp.array(obs["player_0"]["team_points"]),
                    "team_wins": jnp.array(obs["player_0"]["team_wins"]),
                    "steps": obs["player_0"]["steps"],
                    "match_steps": obs["player_0"]["match_steps"],
                    "relic_nodes": jnp.array(obs["player_0"]["relic_nodes"]),
                    "relic_nodes_mask": jnp.array(obs["player_0"]["relic_nodes_mask"])
                },
                "player_1": {
                    "units": {
                        "position": jnp.array(obs["player_1"]["units"]["position"]),
                        "energy": jnp.array(obs["player_1"]["units"]["energy"])
                    },
                    "units_mask": jnp.array(obs["player_1"]["units_mask"]),
                    "map_features": {
                        "energy": jnp.array(obs["player_1"]["map_features"]["energy"]),
                        "tile_type": jnp.array(obs["player_1"]["map_features"]["tile_type"])
                    },
                    "sensor_mask": jnp.array(obs["player_1"]["sensor_mask"]),
                    "team_points": jnp.array(obs["player_1"]["team_points"]),
                    "team_wins": jnp.array(obs["player_1"]["team_wins"]),
                    "steps": obs["player_1"]["steps"],
                    "match_steps": obs["player_1"]["match_steps"],
                    "relic_nodes": jnp.array(obs["player_1"]["relic_nodes"]),
                    "relic_nodes_mask": jnp.array(obs["player_1"]["relic_nodes_mask"])
                }
            })
            episode_actions.append(current_actions)
            
            # Step environment
            rng, step_rng = jax.random.split(rng)
            step_result = env.step(step_rng, state, actions, params)
            raw_obs, state, rewards, done_flags = step_result[:4]  # Unpack only what we need
            
            # Get observation dictionary and convert to policy format
            obs_dict = env.get_obs(state, params)
            obs = convert_obs_to_dict(obs_dict)
            
            # Update reward based on team points, unit counts, and exploration for current player
            current_team_points = float(obs[current_player]["team_points"][current_team_idx])
            current_unit_count = float(np.sum(obs[current_player]["units_mask"]))
            
            # Calculate exploration bonus based on unit positions
            unit_positions = np.array(obs[current_player]["units"]["position"][current_team_idx])
            valid_mask = np.array(obs[current_player]["units_mask"][current_team_idx])
            valid_positions = unit_positions[valid_mask]
            # Convert positions to tuples for set operation
            position_tuples = [tuple(pos) for pos in valid_positions]
            unique_positions = len(set(position_tuples))
            exploration_bonus = 0.05 * unique_positions  # Small bonus for exploring unique positions
            
            # Balance between points, units, and exploration
            current_reward = current_team_points + 0.1 * current_unit_count + exploration_bonus
            episode_reward += current_reward
            
            # Log reward components for debugging
            logging.info(f"Step {step_count} - Team points: {current_team_points:.2f}")
            logging.info(f"Step {step_count} - Unit count contribution: {0.1 * current_unit_count:.2f}")
            logging.info(f"Step {step_count} - Total reward: {current_reward:.2f}")
            
            # Log step information
            if step_count % 10 == 0:
                logging.info(f"Step {step_count} - Active units: {np.sum(obs[current_player]['units_mask'])}")
                logging.info(f"Step {step_count} - Current reward: {current_reward:.2f}")
                logging.info(f"Step {step_count} - Training as: {current_player}")
            
            # Check termination
            done = jnp.any(done_flags[current_player])
        
        # Store episode data
        total_rewards.append(episode_reward)
        step_rewards = [episode_reward / len(episode_observations)] * len(episode_observations)
        
        # Add episode data to overall buffers
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        all_rewards.extend(step_rewards)
        
        # Clear episode buffers
        episode_observations = []
        episode_actions = []
        
        # Update policy if we have enough experience
        if len(all_observations) >= buffer_size:
            # Stack observations into batched format
            obs_batch = create_batched_obs(all_observations)
            action_array = jnp.array(all_actions)  # Shape: [batch_size, max_units]
            reward_array = jnp.array(all_rewards)  # Shape: [batch_size]
            
            # Log shapes for debugging
            logging.info(f"Batch shapes before update:")
            logging.info(f"Action array shape: {action_array.shape}")
            logging.info(f"Reward array shape: {reward_array.shape}")
            for player in ["player_0", "player_1"]:
                logging.info(f"{player} observation shapes:")
                for key, value in obs_batch[player].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            logging.info(f"  {key}.{subkey}: {subvalue.shape}")
                    else:
                        logging.info(f"  {key}: {value.shape}")
            
            # Normalize rewards
            reward_array = (reward_array - reward_array.mean()) / (reward_array.std() + 1e-8)
            
            # Log pre-update state
            logging.info("Pre-update policy state:")
            params_dict = flax.traverse_util.flatten_dict(policy_state.params)
            for key, value in params_dict.items():
                if any(layer in '.'.join(key) for layer in ['Dense_0', 'Dense_1', 'Dense_2']):
                    logging.info(f"{'.'.join(key)} mean: {jnp.mean(value)}")
            
            # Update policy
            policy_state, loss = update_step(
                policy, policy_state,
                obs_batch, action_array, reward_array,
                optimizer
            )
            
            # Log post-update state
            logging.info("Post-update policy state:")
            params_dict = flax.traverse_util.flatten_dict(policy_state.params)
            for key, value in params_dict.items():
                if any(layer in '.'.join(key) for layer in ['Dense_0', 'Dense_1', 'Dense_2']):
                    logging.info(f"{'.'.join(key)} mean: {jnp.mean(value)}")
            
            logging.info(f"Update loss: {float(loss)}")
            logging.info(f"Reward stats - mean: {float(jnp.mean(reward_array))}, std: {float(jnp.std(reward_array))}")
            episode_losses.append(float(loss))
            
            # Clear buffers
            all_observations = []
            all_actions = []
            all_rewards = []
        
        # Log progress
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(total_rewards[-10:])
            logging.info("-" * 40)
            logging.info(f"Episode {episode + 1}/{num_episodes}")
            logging.info(f"Mean reward (last 10): {mean_reward:.2f}")
            logging.info(f"Latest episode reward: {episode_reward:.2f}")
            if episode_losses:
                logging.info(f"Latest loss: {episode_losses[-1]:.4f}")
                logging.info(f"Total policy updates: {len(episode_losses)}")
            logging.info(f"Total steps: {step_count}")
            logging.info(f"Buffer size: {len(all_observations)}/{buffer_size}")
            logging.info("-" * 40)
    
    # Save trained policy parameters
    # Convert policy parameters to flat numpy arrays
    # Extract kernel and bias from each layer
    policy_params_dict = flax.traverse_util.flatten_dict(policy_state.params)
    numpy_params = {}
    for key, value in policy_params_dict.items():
        numpy_params['.'.join(str(k) for k in key)] = np.array(value)
    
    # Save parameters as individual arrays
    save_path = os.path.join(os.path.dirname(__file__), "model_params.npz")
    save_dict = {
        **numpy_params,
        'mean_reward': np.array(np.mean(total_rewards)),
        'hidden_dims': np.array(policy.hidden_dims),
        'max_units': np.array(params.max_units),
        'num_actions': np.array(5)
    }
    np.savez(save_path, **save_dict)
    logging.info(f"Saved model parameters to {save_path}")
    logging.info(f"Training complete. Mean reward: {np.mean(total_rewards):.2f}")
    logging.info("Saved model parameters to kits/rl/model_params.npz")
def convert_obs_to_dict(raw_obs: Union[EnvObs, Dict[str, EnvObs]]) -> Dict[str, Dict[str, Any]]:
    """Convert raw observation to policy network format.
    
    Args:
        raw_obs: Either a single EnvObs object or a dictionary mapping player to EnvObs
    """
    obs_dict = {}
    
    def _convert_single_obs(obs: EnvObs) -> Dict[str, Any]:
        """Convert a single EnvObs to dictionary format."""
        return {
            "units": {
                "position": jnp.array(obs.units.position),
                "energy": jnp.array(obs.units.energy)
            },
            "units_mask": jnp.array(obs.units_mask),
            "map_features": {
                "energy": jnp.array(obs.map_features.energy),
                "tile_type": jnp.array(obs.map_features.tile_type)
            },
            "sensor_mask": jnp.array(obs.sensor_mask),
            "team_points": jnp.array(obs.team_points),
            "team_wins": jnp.array(obs.team_wins),
            "steps": obs.steps,
            "match_steps": obs.match_steps,
            "relic_nodes": jnp.array(obs.relic_nodes),
            "relic_nodes_mask": jnp.array(obs.relic_nodes_mask)
        }
    
    if isinstance(raw_obs, dict):
        # Handle Dict[str, EnvObs] case
        for player in ["player_0", "player_1"]:
            obs_dict[player] = _convert_single_obs(raw_obs[player])
    else:
        # Handle single EnvObs case
        obs_dict["player_0"] = _convert_single_obs(raw_obs)
        obs_dict["player_1"] = _convert_single_obs(raw_obs)
    
    return obs_dict

def create_batched_obs(observations: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Create batched observations from a list of dictionary format observations."""
    return {
        player: {
            "units": {
                "position": jnp.stack([obs[player]["units"]["position"] for obs in observations]),
                "energy": jnp.stack([obs[player]["units"]["energy"] for obs in observations])
            },
            "units_mask": jnp.stack([obs[player]["units_mask"] for obs in observations]),
            "map_features": {
                "energy": jnp.stack([obs[player]["map_features"]["energy"] for obs in observations]),
                "tile_type": jnp.stack([obs[player]["map_features"]["tile_type"] for obs in observations])
            },
            "sensor_mask": jnp.stack([obs[player]["sensor_mask"] for obs in observations]),
            "team_points": jnp.stack([obs[player]["team_points"] for obs in observations]),
            "team_wins": jnp.stack([obs[player]["team_wins"] for obs in observations]),
            "steps": jnp.array([obs[player]["steps"] for obs in observations]),
            "match_steps": jnp.array([obs[player]["match_steps"] for obs in observations]),
            "relic_nodes": jnp.stack([obs[player]["relic_nodes"] for obs in observations]),
            "relic_nodes_mask": jnp.stack([obs[player]["relic_nodes_mask"] for obs in observations])
        }
        for player in ["player_0", "player_1"]
    }

if __name__ == "__main__":
    train_basic_env()
