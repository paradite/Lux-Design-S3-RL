import json
import os
import sys
import numpy as np
from argparse import Namespace
from lux.kit import from_json

class TrainedAgent:
    def __init__(self, player: str, env_cfg) -> None:
        """Initialize the trained RL agent"""
        self.player = player
        self.opp_player = "player_1" if player == "player_0" else "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        
        # Load trained parameters
        try:
            data = np.load(os.path.join(os.path.dirname(__file__), "model_params.npz"))
            self.weights = data["dummy_weights"]
            self.mean_reward = data.get("mean_reward", 0.0)
        except Exception as e:
            print(f"Warning: Could not load model parameters: {e}", file=sys.stderr)
            self.weights = np.zeros((1,))
            self.mean_reward = 0.0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Generate actions for all units based on current observation"""
        # Get unit mask to know which units we can control
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        
        # Initialize actions array for all units
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)
        
        # For each available unit, generate a basic action
        # Action format: [movement (0-4), sap_direction_x, sap_direction_y]
        for unit_id in available_unit_ids:
            # For now, just move randomly as a demonstration
            # Generate random actions as numpy arrays
            actions[unit_id] = np.array([
                np.random.randint(0, 5),  # Random movement
                0,  # No sap x direction
                0   # No sap y direction
            ], dtype=np.int32)
        
        # Ensure actions are numpy arrays and properly formatted
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
