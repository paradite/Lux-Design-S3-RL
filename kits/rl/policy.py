import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.core
import optax
from typing import Sequence, Dict, Any, List
from luxai_s3.state import EnvObs, UnitState, MapTile
from flax import struct
import chex

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
    def __call__(self, obs: EnvObs, team_idx: int = 0):
        """Process observation into action logits.
        
        Args:
            obs: EnvObs containing the current observation
            team_idx: Index for the current team (0 or 1)
        """
        # Get current player's observation
        player_obs = obs
        
        # Extract relevant features from observation
        # Get position and energy from UnitState dataclass
        units_pos = player_obs.units.position  # Shape: (2, max_units, 2)
        units_energy = player_obs.units.energy  # Shape: (2, max_units)
        
        # Get unit mask directly from EnvObs dataclass
        units_mask = player_obs.units_mask  # Shape: (2, max_units)
        
        # Get current team's features using team index
        team_pos = jnp.take(units_pos, team_idx, axis=0)    # Shape: (max_units, 2)
        team_energy = jnp.take(units_energy, team_idx, axis=0)  # Shape: (max_units,)
        team_mask = jnp.take(units_mask, team_idx, axis=0)  # Shape: (max_units,)
        
        # Process each unit independently
        x = jnp.concatenate([
            team_pos,  # Shape: (max_units, 2)
            jnp.expand_dims(team_energy, axis=-1),  # Shape: (max_units, 1)
            jnp.expand_dims(team_mask.astype(jnp.float32), axis=-1)  # Shape: (max_units, 1)
        ], axis=-1)  # Final shape: (max_units, 4)
        
        # Simple feedforward network
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Output layer for action logits
        action_logits = nn.Dense(self.num_actions)(x)  # Shape: (max_units, num_actions)
        
        # Reshape mask for broadcasting
        mask = team_mask[..., None]  # Shape: (max_units, 1)
        mask = jnp.broadcast_to(mask, action_logits.shape)  # Shape: (max_units, num_actions)
        
        # Mask invalid units
        masked_logits = jnp.where(
            mask,
            action_logits,
            jnp.full_like(action_logits, -1e9)  # Large negative number for masked units
        )
        
        return masked_logits

def create_dummy_obs(max_units=16):
    """Create a dummy observation for initialization."""
    # Create dummy arrays with proper shapes and types
    # Create dummy arrays with proper shapes and types
    position = jnp.zeros((2, max_units, 2), dtype=jnp.int16)  # (teams, units, coords)
    energy = jnp.zeros((2, max_units), dtype=jnp.int16)  # (teams, units)
    
    # Create nested structures using struct.field
    @struct.dataclass
    class DummyUnitState:
        position: chex.Array = struct.field(default_factory=lambda: position)
        energy: chex.Array = struct.field(default_factory=lambda: energy)
    
    @struct.dataclass
    class DummyMapTile:
        energy: chex.Array = struct.field(default_factory=lambda: jnp.zeros((24, 24), dtype=jnp.int16))
        tile_type: chex.Array = struct.field(default_factory=lambda: jnp.zeros((24, 24), dtype=jnp.int16))
    
    @struct.dataclass
    class DummyEnvObs:
        units: UnitState = struct.field(default_factory=DummyUnitState)
        units_mask: chex.Array = struct.field(default_factory=lambda: jnp.zeros((2, max_units), dtype=jnp.bool_))
        map_features: MapTile = struct.field(default_factory=DummyMapTile)
        sensor_mask: chex.Array = struct.field(default_factory=lambda: jnp.zeros((2, 24, 24), dtype=jnp.bool_))
        team_points: chex.Array = struct.field(default_factory=lambda: jnp.zeros((2,), dtype=jnp.int32))
        team_wins: chex.Array = struct.field(default_factory=lambda: jnp.zeros((2,), dtype=jnp.int32))
        steps: int = struct.field(default=0)
        match_steps: int = struct.field(default=0)
        relic_nodes: chex.Array = struct.field(default_factory=lambda: jnp.zeros((6, 2), dtype=jnp.int16))
        relic_nodes_mask: chex.Array = struct.field(default_factory=lambda: jnp.zeros((6,), dtype=jnp.bool_))
    
    # Create single EnvObs instance
    return DummyEnvObs()

def create_policy(rng, hidden_dims=(64, 64), max_units=16, learning_rate=1e-3):
    """Create and initialize the policy network and optimizer."""
    policy = PolicyNetwork(hidden_dims=hidden_dims)
    # Initialize with dummy observation
    dummy_obs = create_dummy_obs(max_units)
    params = policy.init(rng, dummy_obs, team_idx=0)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create initial policy state
    policy_state = PolicyState(params=params, opt_state=opt_state)
    
    return policy, policy_state, optimizer

def sample_action(policy, params, obs: EnvObs, rng, team_idx: int = 0):
    """Sample actions from the policy for all units.
    
    Args:
        policy: PolicyNetwork instance
        params: Policy parameters
        obs: EnvObs containing the current observation
        rng: JAX random key
        team_idx: Index for the current team (0 or 1)
    
    Returns:
        Array of actions for each unit
    """
    logits = policy.apply(params, obs, team_idx)
    # Add small noise for exploration
    noise = jax.random.gumbel(rng, logits.shape)
    actions = jnp.argmax(logits + noise, axis=-1)
    # Mask actions for invalid units using team-specific mask
    return jnp.where(obs.units_mask[team_idx], actions, 0)

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
    
    # Get action logits for team 0
    logits = policy.apply(params, batch_obs, team_idx=0)  # [batch_size, max_units, num_actions]
    
    # Compute log probabilities
    action_probs = jax.nn.softmax(logits)
    selected_probs = jnp.take_along_axis(
        action_probs,
        action_batch[..., None],  # Add dimension for gather
        axis=-1
    )[..., 0]  # Remove gathered dimension
    
    # Mask out invalid units using team 0's mask
    valid_mask = batch_obs.units_mask[0]  # Get team 0's mask
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
