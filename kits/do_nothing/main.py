import json
import sys
from argparse import Namespace
from agent import Agent
from lux.kit import from_json

agent_dict = {}

def agent_fn(observation, configurations):
    """
    Main agent function that handles initialization and calling appropriate agent actions
    """
    global agent_dict
    obs = observation.obs
    if isinstance(obs, str):
        obs = json.loads(obs)
    
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"])
    
    agent = agent_dict[player]
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
