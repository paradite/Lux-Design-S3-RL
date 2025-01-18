import numpy as np

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if player == "player_0" else "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Returns zero actions for all units, effectively doing nothing.
        
        Args:
            step: Current game step
            obs: Game observation
            remainingOverageTime: Remaining time for computation
            
        Returns:
            numpy array of shape (max_units, 3) containing all zeros
        """
        # Get unit mask to know which units we can control
        unit_mask = np.array(obs["units_mask"][self.team_id])
        
        # Initialize actions array for all units
        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=np.int32)  # Match RL bot's dtype
        
        return actions
