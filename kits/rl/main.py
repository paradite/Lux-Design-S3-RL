import json
import os
import sys
import numpy as np
import logging
from argparse import Namespace
from lux.kit import from_json

import jax
import jax.numpy as jnp
from flax import struct
from policy import PolicyNetwork, create_dummy_obs
from luxai_s3.state import EnvObs, UnitState, MapTile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)

class TrainedAgent:
    def __init__(self, player: str, env_cfg) -> None:
        """Initialize the trained RL agent with policy network"""
        self.player = player
        self.opp_player = "player_1" if player == "player_0" else "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        
        # Initialize random key for JAX
        self.key = jax.random.PRNGKey(0)
        
        # Load trained parameters and initialize policy
        try:
            data = np.load(os.path.join(os.path.dirname(__file__), "model_params.npz"))
            self.policy_params = jax.tree_map(jnp.array, data["policy_params"])
            self.mean_reward = data.get("mean_reward", 0.0)
            self.hidden_dims = tuple(data["hidden_dims"])
            
            # Initialize policy network
            self.policy = PolicyNetwork(hidden_dims=self.hidden_dims)
            
            logging.info(f"Successfully loaded model for player {player}")
            logging.info(f"Model architecture: hidden_dims={self.hidden_dims}")
            logging.info(f"Previous mean reward: {self.mean_reward:.2f}")
            
        except Exception as e:
            logging.error(f"Could not load model parameters: {e}")
            # Initialize with dummy parameters
            dummy_obs = create_dummy_obs()
            self.policy = PolicyNetwork(hidden_dims=(64, 64))
            self.key, init_key = jax.random.split(self.key)
            self.policy_params = self.policy.init(init_key, dummy_obs)
            self.mean_reward = 0.0
            self.hidden_dims = (64, 64)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Generate actions using the trained policy network"""
        # Get unit mask to know which units we can control
        unit_mask = np.array(obs["units_mask"][self.team_id])
        
        # Initialize actions array for all units
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)
        
        # If policy failed to load, fall back to random actions
        if self.policy is None:
            for unit_id in np.where(unit_mask)[0]:
                actions[unit_id] = np.array([
                    np.random.randint(0, 5), 0, 0
                ], dtype=np.int32)
            return actions
        
        # Create proper observation structure using struct.replace
        empty_unit_state = UnitState()
        # Create unit state with position and energy for current team
        position = jnp.array(obs["units"][self.team_id]["position"], dtype=jnp.int16)  # Shape: (max_units, 2)
        energy = jnp.array(obs["units"][self.team_id]["energy"], dtype=jnp.int16)  # Shape: (max_units,)
        
        # Create unit state with proper shapes
        unit_state = struct.replace(
            empty_unit_state,
            position=position,  # Shape: (max_units, 2)
            energy=energy  # Shape: (max_units,)
        )
        
        empty_map_tile = MapTile()
        map_features = struct.replace(
            empty_map_tile,
            energy=jnp.array(obs["map_features"]["energy"], dtype=jnp.int16),
            tile_type=jnp.array(obs["map_features"]["tile_type"], dtype=jnp.int16)
        )
        
        # Create EnvObs object using struct.replace
        empty_env_obs = EnvObs()
        env_obs = struct.replace(
            empty_env_obs,
            units=unit_state,
            units_mask=jnp.array(unit_mask, dtype=jnp.bool_),
            map_features=map_features,
            sensor_mask=jnp.array(obs["sensor_mask"], dtype=jnp.bool_),
            team_points=jnp.array(obs["team_points"], dtype=jnp.int32),
            team_wins=jnp.array(obs["team_wins"], dtype=jnp.int32),
            steps=step,
            match_steps=step,
            relic_nodes=jnp.array(obs["relic_nodes"], dtype=jnp.int16),
            relic_nodes_mask=jnp.array(obs["relic_nodes_mask"], dtype=jnp.bool_)
        )
        
        # Wrap in dictionary for policy
        policy_obs = {self.player: env_obs}
        
        # Generate action logits with exploration noise
        self.key, action_key = jax.random.split(self.key)
        logits = self.policy.apply(self.policy_params, policy_obs, self.player)
        noise = jax.random.gumbel(action_key, logits.shape)
        movement_actions = np.array(jnp.argmax(logits + noise, axis=-1))
        
        # Log action distribution and unit status
        action_counts = np.bincount(movement_actions[unit_mask], minlength=5)
        active_units = np.sum(unit_mask)
        logging.debug(f"Step {step} - Action distribution: {action_counts}")
        logging.debug(f"Step {step} - Active units: {active_units}")
        
        # Set movement actions for valid units using numpy indexing
        actions[:, 0] = movement_actions
        
        # Ensure actions are properly formatted
        return np.array(actions, dtype=np.int32)

# Global dictionary to store agent instances
agent_dict = {}

def agent_fn(observation, configurations):
    """
    Main agent function that handles initialization and calling appropriate agent actions
    """
    global agent_dict
    
    # Parse observation
    obs = observation.obs
    if isinstance(obs, str):
        obs = json.loads(obs)
    
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    
    # Initialize agent if this is the first step
    if step == 0:
        agent_dict[player] = TrainedAgent(player, configurations["env_cfg"])
    
    # Get agent instance
    agent = agent_dict[player]
    
    # Get actions from agent
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    
    return dict(action=actions.tolist())

if __name__ == "__main__":
    def read_input():
        """Read input from stdin"""
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        
        observation = Namespace(**dict(
            step=raw_input["step"],
            obs=raw_input["obs"],
            remainingOverageTime=raw_input["remainingOverageTime"],
            player=raw_input["player"],
            info=raw_input["info"]
        ))
        
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        
        # Send actions to engine
        print(json.dumps(actions))
