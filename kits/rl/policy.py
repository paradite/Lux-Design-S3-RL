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
    def __call__(self, obs: Dict[str, EnvObs], player_key: str = "player_0"):
        """Process observation into action logits.
        
        Args:
            obs: Dict[str, EnvObs] containing observations for each player
            player_key: Key for the current player's observation
        """
        # Get current player's observation
        player_obs = obs[player_key]
        
        # Extract relevant features from player's observation
        # Get position and energy from UnitState
        units_pos = player_obs.units.position  # Shape: (max_units, 2)
        units_energy = player_obs.units.energy  # Shape: (max_units,)
        
        # Get unit mask
        units_mask = player_obs.units_mask  # Shape: (max_units,)
        
        # Process each unit independently
        x = jnp.concatenate([
            units_pos,  # Shape: (max_units, 2)
            jnp.expand_dims(units_energy, axis=1),  # Shape: (max_units, 1)
            jnp.expand_dims(units_mask, axis=1).astype(jnp.float32)  # Shape: (max_units, 1)
        ], axis=-1)  # Final shape: (max_units, 4)
        
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
    dummy_energy = jnp.zeros((max_units,), dtype=jnp.int16)  # Shape should match UnitState
    dummy_units_mask = jnp.zeros((max_units,), dtype=jnp.bool_)
    dummy_map_energy = jnp.zeros((24, 24), dtype=jnp.int16)
    dummy_map_tile_type = jnp.zeros((24, 24), dtype=jnp.int16)
    dummy_sensor_mask = jnp.zeros((24, 24), dtype=jnp.bool_)
    dummy_team_points = jnp.zeros((2,), dtype=jnp.int32)
    dummy_team_wins = jnp.zeros((2,), dtype=jnp.int32)
    dummy_relic_nodes = jnp.zeros((6, 2), dtype=jnp.int16)
    dummy_relic_nodes_mask = jnp.zeros((6,), dtype=jnp.bool_)
    
    # Create nested structures using struct.field
    @struct.dataclass
    class DummyUnitState:
        position: chex.Array = struct.field(default_factory=lambda: dummy_position)
        energy: chex.Array = struct.field(default_factory=lambda: dummy_energy)
    
    @struct.dataclass
    class DummyMapTile:
        energy: chex.Array = struct.field(default_factory=lambda: dummy_map_energy)
        tile_type: chex.Array = struct.field(default_factory=lambda: dummy_map_tile_type)
    
    @struct.dataclass
    class DummyEnvObs:
        units: UnitState = struct.field(default_factory=lambda: DummyUnitState())
        units_mask: chex.Array = struct.field(default_factory=lambda: dummy_units_mask)
        map_features: MapTile = struct.field(default_factory=lambda: DummyMapTile())
        sensor_mask: chex.Array = struct.field(default_factory=lambda: dummy_sensor_mask)
        team_points: chex.Array = struct.field(default_factory=lambda: dummy_team_points)
        team_wins: chex.Array = struct.field(default_factory=lambda: dummy_team_wins)
        steps: int = struct.field(default=0)
        match_steps: int = struct.field(default=0)
        relic_nodes: chex.Array = struct.field(default_factory=lambda: dummy_relic_nodes)
        relic_nodes_mask: chex.Array = struct.field(default_factory=lambda: dummy_relic_nodes_mask)
    
    # Create dummy observations for both players
    dummy_obs = DummyEnvObs()
    return {
        "player_0": dummy_obs,
        "player_1": dummy_obs
    }

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

def sample_action(policy, params, obs: Dict[str, EnvObs], rng, player_key: str = "player_0"):
    """Sample actions from the policy for all units.
    
    Args:
        policy: PolicyNetwork instance
        params: Policy parameters
        obs: Dict[str, EnvObs] containing observations for each player
        rng: JAX random key
        player_key: Key for the current player's observation
    
    Returns:
        Array of actions for each unit
    """
    logits = policy.apply(params, obs, player_key)
    # Add small noise for exploration
    noise = jax.random.gumbel(rng, logits.shape)
    actions = jnp.argmax(logits + noise, axis=-1)
    # Mask actions for invalid units
    return jnp.where(obs[player_key].units_mask, actions, 0)

def compute_loss(policy, params, obs_batch: List[Dict[str, EnvObs]], action_batch, reward_batch):
    """Compute policy gradient loss.
    
    Args:
        policy: PolicyNetwork instance
        params: Policy parameters
        obs_batch: List of Dict[str, EnvObs] objects
        action_batch: Array of actions taken (shape: [batch_size, max_units])
        reward_batch: Array of rewards (shape: [batch_size])
    
    Returns:
        Scalar loss value
    """
    # Stack observations into a single batch
    batch_obs = jax.tree_map(lambda *x: jnp.stack(x), *obs_batch)
    
    # Get action logits for player_0
    logits = policy.apply(params, batch_obs, "player_0")  # [batch_size, max_units, num_actions]
    
    # Compute log probabilities
    action_probs = jax.nn.softmax(logits)
    selected_probs = jnp.take_along_axis(
        action_probs,
        action_batch[..., None],  # Add dimension for gather
        axis=-1
    )[..., 0]  # Remove gathered dimension
    
    # Mask out invalid units using player_0's mask
    valid_mask = batch_obs["player_0"].units_mask
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
