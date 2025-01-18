import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import logging
from typing import Sequence, Dict, Any, List, Optional
from luxai_s3.state import EnvObs, UnitState, MapTile
from flax import struct
import flax.core
import chex

@struct.dataclass
class PolicyState:
    """State of the policy network."""
    params: Any
    opt_state: Any

class PolicyNetwork(nn.Module):
    """Simple policy network for the Lux AI Season 3 environment."""
    hidden_dims: Sequence[int]
    num_actions: int = 5  # 0-4 for movement
    
    @nn.compact
    def __call__(self, obs: Dict[str, Dict], player: str):
        """Process observation into action logits.
        
        Args:
            obs: Dictionary mapping player to their observation dictionary
            player: Current player ("player_0" or "player_1")
        """
        # Get current player's observation
        player_obs = obs[player]
        team_idx = jnp.array(0 if player == "player_0" else 1, dtype=jnp.int32)
        
        # Extract relevant features from dictionary and get current team's data
        # Handle both batched and unbatched inputs
        units_pos = jnp.array(player_obs["units"]["position"], dtype=jnp.int16)
        units_energy = jnp.array(player_obs["units"]["energy"], dtype=jnp.int16)
        units_mask = jnp.array(player_obs["units_mask"], dtype=jnp.bool_)
        
        # Handle different input shapes
        if len(units_pos.shape) == 4:  # Batched with team dim: (batch, 2, max_units, 2)
            units_pos = units_pos[:, team_idx]  # Shape: (batch, max_units, 2)
            units_energy = units_energy[:, team_idx]  # Shape: (batch, max_units)
            units_mask = units_mask[:, team_idx]  # Shape: (batch, max_units)
        elif len(units_pos.shape) == 3:  # Batched without team dim or unbatched with team dim
            if units_pos.shape[0] == 2:  # Unbatched with team dim: (2, max_units, 2)
                units_pos = jnp.expand_dims(units_pos[team_idx], axis=0)  # Shape: (1, max_units, 2)
                units_energy = jnp.expand_dims(units_energy[team_idx], axis=0)  # Shape: (1, max_units)
                units_mask = jnp.expand_dims(units_mask[team_idx], axis=0)  # Shape: (1, max_units)
            else:  # Batched without team dim: (batch, max_units, 2)
                pass  # Already in correct shape
        else:  # Unbatched without team dim: (max_units, 2)
            units_pos = jnp.expand_dims(units_pos, axis=0)  # Shape: (1, max_units, 2)
            units_energy = jnp.expand_dims(units_energy, axis=0)  # Shape: (1, max_units)
            units_mask = jnp.expand_dims(units_mask, axis=0)  # Shape: (1, max_units)
        
        # Reshape and normalize features for network input
        # Convert to float32 and normalize position to [-1,1] range to make it symmetric for both players
        pos_feature = (units_pos.astype(jnp.float32) / 12.0) - 1.0  # Shape: (batch_size, max_units, 2)
        
        # Convert energy to float32 and normalize to [0,1] range
        energy_feature = jnp.expand_dims(units_energy.astype(jnp.float32) / 100.0, axis=-1)  # Shape: (batch_size, max_units, 1)
        
        # Convert mask to float32 (already binary 0/1)
        mask_feature = jnp.expand_dims(units_mask.astype(jnp.float32), axis=-1)  # Shape: (batch_size, max_units, 1)
        
        # Create team index feature (already binary 0/1)
        team_idx_feature = jnp.ones_like(mask_feature, dtype=jnp.float32) * team_idx.astype(jnp.float32)  # Shape: (batch_size, max_units, 1)
        
        # Only log feature info in debug mode
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            logging.debug(f"Feature shapes and dtypes:")
            logging.debug(f"  pos_feature: shape={pos_feature.shape}, dtype={pos_feature.dtype}")
            logging.debug(f"  energy_feature: shape={energy_feature.shape}, dtype={energy_feature.dtype}")
            logging.debug(f"  mask_feature: shape={mask_feature.shape}, dtype={mask_feature.dtype}")
            logging.debug(f"  team_idx_feature: shape={team_idx_feature.shape}, dtype={team_idx_feature.dtype}")
        
        x = jnp.concatenate([
            pos_feature,  # Shape: (batch_size, max_units, 2)
            energy_feature,  # Shape: (batch_size, max_units, 1)
            mask_feature,  # Shape: (batch_size, max_units, 1)
            team_idx_feature  # Shape: (batch_size, max_units, 1)
        ], axis=-1)  # Final shape: (batch_size, max_units, 5)
        
        # Only log in debug mode
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            logging.debug(f"Concatenated feature tensor x: shape={x.shape}, dtype={x.dtype}")
            logging.debug(f"Feature value ranges:")
            logging.debug(f"  pos_feature: min={jnp.min(pos_feature)}, max={jnp.max(pos_feature)}")
            logging.debug(f"  energy_feature: min={jnp.min(energy_feature)}, max={jnp.max(energy_feature)}")
            logging.debug(f"  mask_feature: min={jnp.min(mask_feature)}, max={jnp.max(mask_feature)}")
            logging.debug(f"  team_idx_feature: min={jnp.min(team_idx_feature)}, max={jnp.max(team_idx_feature)}")
        
        # Process each unit's features through the network
        batch_size = x.shape[0]
        max_units = x.shape[1]
        
        # Reshape for dense layers while preserving batch structure
        x = x.reshape(-1, x.shape[-1])  # Shape: (batch_size * max_units, 5)
        
        # Simple feedforward network
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Output layer for action logits
        action_logits = nn.Dense(self.num_actions)(x)  # Shape: (batch_size * max_units, num_actions)
        
        # Reshape back to include batch and unit dimensions
        action_logits = action_logits.reshape(batch_size, max_units, self.num_actions)
        
        # Reshape mask for broadcasting
        mask = jnp.broadcast_to(mask_feature, action_logits.shape)
        
        # Mask invalid units
        masked_logits = jnp.where(
            mask > 0,  # mask_feature is float32, so compare with 0
            action_logits,
            jnp.full_like(action_logits, -1e9)  # Large negative number for masked units
        )
        
        return masked_logits

def create_dummy_obs(max_units=16):
    """Create a dummy observation dictionary for initialization."""
    # Create dummy observation in raw dictionary format
    dummy_obs = {
        "units": {
            "position": jnp.zeros((max_units, 2), dtype=jnp.int16),
            "energy": jnp.zeros((max_units,), dtype=jnp.int16)
        },
        "units_mask": jnp.zeros((max_units,), dtype=jnp.bool_),
        "map_features": {
            "energy": jnp.zeros((24, 24), dtype=jnp.int16),
            "tile_type": jnp.zeros((24, 24), dtype=jnp.int16)
        },
        "sensor_mask": jnp.zeros((24, 24), dtype=jnp.bool_),
        "team_points": jnp.zeros((2,), dtype=jnp.int32),
        "team_wins": jnp.zeros((2,), dtype=jnp.int32),
        "steps": 0,
        "match_steps": 0,
        "relic_nodes": jnp.zeros((6, 2), dtype=jnp.int16),
        "relic_nodes_mask": jnp.zeros((6,), dtype=jnp.bool_)
    }
    return dummy_obs

def create_policy(rng, hidden_dims=(128, 128, 64), max_units=16, learning_rate=1e-3):
    """Create and initialize the policy network and optimizer."""
    policy = PolicyNetwork(hidden_dims=hidden_dims)
    
    # Initialize with dummy observation dictionary in raw format
    dummy_obs = {
        "player_0": {
            "units": {
                "position": jnp.zeros((2, max_units, 2), dtype=jnp.int16),
                "energy": jnp.zeros((2, max_units), dtype=jnp.int16)
            },
            "units_mask": jnp.zeros((2, max_units), dtype=jnp.bool_),
            "map_features": {
                "energy": jnp.zeros((24, 24), dtype=jnp.int16),
                "tile_type": jnp.zeros((24, 24), dtype=jnp.int16)
            },
            "sensor_mask": jnp.zeros((24, 24), dtype=jnp.bool_),
            "team_points": jnp.zeros((2,), dtype=jnp.int32),
            "team_wins": jnp.zeros((2,), dtype=jnp.int32),
            "steps": 0,
            "match_steps": 0,
            "relic_nodes": jnp.zeros((6, 2), dtype=jnp.int16),
            "relic_nodes_mask": jnp.zeros((6,), dtype=jnp.bool_)
        },
        "player_1": {
            "units": {
                "position": jnp.zeros((2, max_units, 2), dtype=jnp.int16),
                "energy": jnp.zeros((2, max_units), dtype=jnp.int16)
            },
            "units_mask": jnp.zeros((2, max_units), dtype=jnp.bool_),
            "map_features": {
                "energy": jnp.zeros((24, 24), dtype=jnp.int16),
                "tile_type": jnp.zeros((24, 24), dtype=jnp.int16)
            },
            "sensor_mask": jnp.zeros((24, 24), dtype=jnp.bool_),
            "team_points": jnp.zeros((2,), dtype=jnp.int32),
            "team_wins": jnp.zeros((2,), dtype=jnp.int32),
            "steps": 0,
            "match_steps": 0,
            "relic_nodes": jnp.zeros((6, 2), dtype=jnp.int16),
            "relic_nodes_mask": jnp.zeros((6,), dtype=jnp.bool_)
        }
    }
    
    # Initialize policy parameters
    params = policy.init(rng, dummy_obs, "player_0")
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create initial policy state
    policy_state = PolicyState(params=params, opt_state=opt_state)
    
    return policy, policy_state, optimizer

def sample_action(policy, policy_state, obs: Dict[str, Dict[str, Any]], player: str, rng):
    """Sample actions from the policy for all units.
    
    Args:
        policy: PolicyNetwork instance
        policy_state: PolicyState instance containing parameters
        obs: Dictionary mapping player to their raw observation dictionary
        player: Current player ("player_0" or "player_1")
        rng: JAX random key
    
    Returns:
        Array of actions for each unit with shape (max_units,)
    """
    logits = policy.apply(policy_state.params, obs, player)  # Shape: (max_units, num_actions)
    # Add small noise for exploration
    noise = jax.random.gumbel(rng, logits.shape)
    actions = jnp.argmax(logits + noise, axis=-1)  # Shape: (max_units,)
    # Mask actions for invalid units using player's mask
    team_idx = jnp.array(0 if player == "player_0" else 1, dtype=jnp.int32)
    valid_mask = jnp.array(obs[player]["units_mask"], dtype=jnp.bool_)[team_idx]
    return jnp.where(valid_mask, actions, 0)  # Shape: (max_units,)

def compute_loss(policy, policy_state, obs_batch: Dict[str, Dict], action_batch, reward_batch):
    """Compute policy gradient loss.
    
    Args:
        policy: PolicyNetwork instance
        policy_state: PolicyState instance containing parameters
        obs_batch: Dictionary mapping player to batched raw observation dictionary
                  containing units, units_mask, etc.
        action_batch: Array of actions taken (shape: [batch_size, max_units])
        reward_batch: Array of rewards (shape: [batch_size])
    
    Returns:
        Scalar loss value
    """
    # Get action logits for player_0
    logits = policy.apply(policy_state.params, obs_batch, "player_0")  # [batch_size, max_units, num_actions]
    
    # Ensure action_batch has correct shape
    action_batch = jnp.array(action_batch)  # Shape: [batch_size, max_units] or [batch_size, 1, max_units]
    if len(action_batch.shape) == 3:
        action_batch = jnp.squeeze(action_batch, axis=1)  # Shape: [batch_size, max_units]
    
    # Get player_0's units mask
    team_idx = jnp.array(0, dtype=jnp.int32)  # player_0 is always team 0 in training
    valid_mask = jnp.array(obs_batch["player_0"]["units_mask"], dtype=jnp.float32)[:, team_idx]  # Shape: [batch_size, max_units]
    
    # Compute log probabilities only for player_0's units
    action_probs = jax.nn.softmax(logits)  # Shape: [batch_size, max_units, num_actions]
    
    # Reshape action_batch to match logits shape for gathering
    action_indices = jnp.expand_dims(action_batch, axis=-1)  # Shape: [batch_size, max_units, 1]
    
    # Gather probabilities of selected actions
    selected_probs = jnp.take_along_axis(
        action_probs,
        action_indices,
        axis=-1
    )[..., 0]  # Shape: [batch_size, max_units]
    
    # Apply mask to log probabilities
    log_probs = jnp.where(
        valid_mask > 0,
        jnp.log(selected_probs + 1e-8),
        0.0
    )
    
    # Sum log probs for each batch element
    batch_log_probs = jnp.sum(log_probs, axis=-1)
    
    # Compute policy gradient loss
    loss = -jnp.mean(batch_log_probs * reward_batch)
    return loss

@functools.partial(jax.jit, static_argnums=(0, 5))
def update_step(policy, policy_state, obs_batch, action_batch, reward_batch, optimizer):
    """Single update step for policy parameters.
    
    Args:
        policy: PolicyNetwork instance (static)
        policy_state: Current policy state with parameters and optimizer state
        obs_batch: Batched observations
        action_batch: Batched actions
        reward_batch: Batched rewards
        optimizer: Optax optimizer
    """
    loss_fn = lambda p: compute_loss(policy, PolicyState(params=p, opt_state=policy_state.opt_state), obs_batch, action_batch, reward_batch)
    loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
    updates, new_opt_state = optimizer.update(grads, policy_state.opt_state)
    new_params = optax.apply_updates(policy_state.params, updates)
    
    # Create new policy state
    new_policy_state = PolicyState(params=new_params, opt_state=new_opt_state)
    return new_policy_state, loss
