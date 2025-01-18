import json
import os
import sys
import numpy as np
import logging
from argparse import Namespace
from lux.kit import from_json

import jax
import jax.numpy as jnp
from policy import PolicyNetwork

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
        
        # Load trained parameters and initialize policy
        try:
            data = np.load(os.path.join(os.path.dirname(__file__), "model_params.npz"))
            self.weights = data["dummy_weights"]  # These are the policy parameters
            self.mean_reward = data.get("mean_reward", 0.0)
            self.hidden_dims = tuple(data["hidden_dims"])
            self.num_actions = int(data.get("num_actions", 5))
            
            # Initialize policy network
            self.policy = PolicyNetwork(
                hidden_dims=self.hidden_dims,
                num_actions=self.num_actions
            )
            
            # Convert numpy arrays to jax arrays for policy
            self.params = jax.tree_map(jnp.array, self.weights)
            
            # Log model loading details
            logging.info(f"Successfully loaded model for player {player}")
            logging.info(f"Model architecture: hidden_dims={self.hidden_dims}")
            logging.info(f"Previous mean reward: {self.mean_reward:.2f}")
            logging.info(f"Action space size: {self.num_actions}")
            
        except Exception as e:
            print(f"Warning: Could not load model parameters: {e}", file=sys.stderr)
            self.weights = np.zeros((1,))
            self.mean_reward = 0.0
            self.hidden_dims = (64, 64)
            self.num_actions = 5
            self.policy = None
            self.params = None

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
        
        # Prepare observation for policy
        policy_obs = {
            "units": {
                "position": jnp.array(obs["units"][self.team_id]["position"]),
                "energy": jnp.array(obs["units"][self.team_id]["energy"])
            },
            "units_mask": jnp.array(unit_mask),
            "map_features": {
                "energy": jnp.array(obs["map_features"]["energy"]),
                "tile_type": jnp.array(obs["map_features"]["tile_type"])
            },
            "sensor_mask": jnp.array(obs["sensor_mask"])
        }
        
        # Get action logits from policy
        logits = self.policy.apply(self.params, policy_obs)
        
        # Convert logits to actions (convert to numpy for final output)
        movement_actions = np.array(jnp.argmax(logits, axis=-1))
        
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
