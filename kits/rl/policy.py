import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.core
import optax
from typing import Sequence, Dict, Any, List
from luxai_s3.state import EnvObs, UnitState, MapTile
from flax import struct

class PolicyState:
    """State of the policy network."""
    def __init__(self, params=None, opt_state=None):
        self.params = params
        self.opt_state = opt_state

class PolicyNetwork(nn.Module):
    """Simple policy network for the Lux AI Season 3 environment."""
    hidden_dims: Sequence[int]
    num_actions: int = 5  # 0-4 for movement
    
    @nn.compact
    def __call__(self, obs: EnvObs):
        """Process observation into action logits.
        
        Args:
            obs: EnvObs containing:
                - units: UnitState with position and energy
                - units_mask: Mask of valid units
                - map_features: MapTile with energy and tile_type
                - sensor_mask: Visibility mask
        """
        # Extract relevant features from EnvObs
        units_pos = obs.units.position  # Shape: (max_units, 2)
        units_energy = obs.units.energy  # Shape: (max_units,)
        units_mask = obs.units_mask  # Shape: (max_units,)
        
        # Combine unit features
        unit_features = jnp.concatenate([
            units_pos,
            jnp.expand_dims(units_energy, axis=1),  # Add channel dimension
            jnp.expand_dims(units_mask, axis=1).astype(jnp.float32)
        ], axis=-1)  # Shape: (max_units, 4)
        
        # Process each unit independently
        x = unit_features
        
        # Simple feedforward network
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Output layer for action logits
        action_logits = nn.Dense(self.num_actions)(x)
        
        # Mask invalid units
        action_logits = jnp.where(
            units_mask[:, None],
            action_logits,
            -1e9  # Large negative number for masked units
        )
        
        return action_logits

def create_dummy_obs(max_units=16):
    """Create a dummy observation for initialization."""
    # Create dummy arrays with proper shapes and types
    dummy_position = jnp.zeros((max_units, 2), dtype=jnp.int16)
    dummy_energy = jnp.zeros((max_units, 1), dtype=jnp.int16)
    dummy_units_mask = jnp.zeros((max_units,), dtype=jnp.bool_)
    dummy_map_energy = jnp.zeros((24, 24), dtype=jnp.int16)
    dummy_map_tile_type = jnp.zeros((24, 24), dtype=jnp.int16)
    dummy_sensor_mask = jnp.zeros((24, 24), dtype=jnp.bool_)
    dummy_team_points = jnp.zeros((2,), dtype=jnp.int32)
    dummy_team_wins = jnp.zeros((2,), dtype=jnp.int32)
    dummy_relic_nodes = jnp.zeros((6, 2), dtype=jnp.int16)
    dummy_relic_nodes_mask = jnp.zeros((6,), dtype=jnp.bool_)
    
    # Create empty instances first
    empty_unit_state = UnitState(position=jnp.zeros((1, 2)), energy=jnp.zeros((1, 1)))
    empty_map_tile = MapTile(energy=jnp.zeros((1, 1)), tile_type=jnp.zeros((1, 1)))
    empty_obs = EnvObs(
        units=empty_unit_state,
        units_mask=jnp.zeros((1,)),
        map_features=empty_map_tile,
        sensor_mask=jnp.zeros((1, 1)),
        team_points=jnp.zeros((2,)),
        team_wins=jnp.zeros((2,)),
        steps=0,
        match_steps=0,
        relic_nodes=jnp.zeros((1, 2)),
        relic_nodes_mask=jnp.zeros((1,))
    )
    
    # Create nested structures using struct.replace
    unit_state = struct.replace(
        empty_unit_state,
        position=dummy_position,
        energy=dummy_energy
    )
    
    map_tile = struct.replace(
        empty_map_tile,
        energy=dummy_map_energy,
        tile_type=dummy_map_tile_type
    )
    
    # Create full observation using struct.replace
    return struct.replace(
        empty_obs,
        units=unit_state,
        units_mask=dummy_units_mask,
        map_features=map_tile,
        sensor_mask=dummy_sensor_mask,
        team_points=dummy_team_points,
        team_wins=dummy_team_wins,
        steps=0,
        match_steps=0,
        relic_nodes=dummy_relic_nodes,
        relic_nodes_mask=dummy_relic_nodes_mask
    )

def create_policy(rng, hidden_dims=(64, 64), max_units=16, learning_rate=1e-3):
    """Create and initialize the policy network and optimizer."""
    policy = PolicyNetwork(hidden_dims=hidden_dims)
    # Initialize with dummy observation
    dummy_obs = create_dummy_obs(max_units)
    params = policy.init(rng, dummy_obs)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create initial policy state
    policy_state = PolicyState(params=params, opt_state=opt_state)
    
    return policy, policy_state, optimizer

def sample_action(policy, params, obs: EnvObs, rng):
    """Sample actions from the policy for all units.
    
    Args:
        policy: PolicyNetwork instance
        params: Policy parameters
        obs: EnvObs containing unit and map information
        rng: JAX random key
    
    Returns:
        Array of actions for each unit
    """
    logits = policy.apply(params, obs)
    # Add small noise for exploration
    noise = jax.random.gumbel(rng, logits.shape)
    actions = jnp.argmax(logits + noise, axis=-1)
    # Mask actions for invalid units
    return jnp.where(obs.units_mask, actions, 0)

def compute_loss(policy, params, obs_batch: List[EnvObs], action_batch, reward_batch):
    """Compute policy gradient loss.
    
    Args:
        policy: PolicyNetwork instance
        params: Policy parameters
        obs_batch: List of EnvObs objects
        action_batch: Array of actions taken (shape: [batch_size, max_units])
        reward_batch: Array of rewards (shape: [batch_size])
    
    Returns:
        Scalar loss value
    """
    # Stack observations into a single batch
    batch_obs = jax.tree_map(lambda *x: jnp.stack(x), *obs_batch)
    
    # Get action logits
    logits = policy.apply(params, batch_obs)  # [batch_size, max_units, num_actions]
    
    # Compute log probabilities
    action_probs = jax.nn.softmax(logits)
    selected_probs = jnp.take_along_axis(
        action_probs,
        action_batch[..., None],  # Add dimension for gather
        axis=-1
    )[..., 0]  # Remove gathered dimension
    
    # Mask out invalid units
    valid_mask = batch_obs.units_mask
    log_probs = jnp.where(
        valid_mask,
        jnp.log(selected_probs + 1e-8),
        0.0
    )
    
    # Sum log probs for each batch element
    batch_log_probs = jnp.sum(log_probs, axis=-1)
    
    # Compute policy gradient loss
    loss = -jnp.mean(batch_log_probs * reward_batch)
    return loss

@jax.jit
def update_step(policy, policy_state, obs_batch, action_batch, reward_batch, optimizer):
    """Single update step for policy parameters."""
    loss_fn = lambda p: compute_loss(policy, p, obs_batch, action_batch, reward_batch)
    loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
    updates, new_opt_state = optimizer.update(grads, policy_state.opt_state)
    new_params = optax.apply_updates(policy_state.params, updates)
    
    # Create new policy state
    new_policy_state = PolicyState(params=new_params, opt_state=new_opt_state)
    return new_policy_state, loss
