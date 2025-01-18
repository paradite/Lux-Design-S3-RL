import subprocess
import json
import time
import sys
import logging
from datetime import datetime

# Set logging level to WARNING for RL bot
logging.getLogger().setLevel(logging.WARNING)

def run_game(rl_bot_path, opponent_path, num_games=3, timeout=120):
    """Run multiple games between RL bot and opponent.
    
    Args:
        rl_bot_path: Path to RL bot
        opponent_path: Path to opponent bot
        num_games: Number of games to run
        timeout: Timeout in seconds for each game
    """
    results = []
    
    for game in range(num_games):
        print(f"\nGame {game + 1}/{num_games}")
        print("-" * 50)
        
        process = None
        start_time = time.time()
        
        try:
            # Run the game
            process = subprocess.Popen(
                ["luxai-s3", rl_bot_path, opponent_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            rewards = None
            game_output = []
            
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Game {game + 1} timed out after {timeout} seconds")
                
                # Check if process has ended
                if process.poll() is not None:
                    break
                
                # Read output with timeout
                if process.stdout is None:
                    raise RuntimeError("Failed to capture stdout")
                
                line = process.stdout.readline()
                if not line:
                    continue
                
                # Store and print progress
                line = line.strip()
                if line:
                    game_output.append(line)
                    # Only print non-debug lines
                    if not any(x in line for x in ["INFO", "DEBUG", "Feature", "tensor"]):
                        print(f"Progress: {line}")
                    
                    # Check for rewards
                    if "Rewards:" in line:
                        rewards = line.strip()
                        break  # Game is done when we get rewards
            
            # Get any remaining output
            try:
                _, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            if rewards:
                try:
                    # Extract integers from numpy array strings
                    rewards_str = rewards.split("Rewards:")[1].strip()
                    # Parse player_0 score
                    p0_score = int(rewards_str.split("'player_0': array(")[1].split(",")[0])
                    # Parse player_1 score
                    p1_score = int(rewards_str.split("'player_1': array(")[1].split(",")[0])
                    
                    rewards_dict = {
                        'player_0': p0_score,
                        'player_1': p1_score
                    }
                    results.append({
                        'game': game + 1,
                        'rewards': rewards_dict,
                        'time': time_taken
                    })
                    print(f"Game {game + 1} completed in {time_taken:.2f}s")
                    print(f"Scores - Player 0: {p0_score}, Player 1: {p1_score}")
                except Exception as e:
                    print(f"Error parsing rewards for game {game + 1}: {str(e)}")
                    print(f"Raw rewards string: {rewards}")
            else:
                print(f"Warning: No rewards found for game {game + 1}")
                print("Last few lines of output:")
                for line in game_output[-5:]:
                    print(line)
        
        except Exception as e:
            print(f"Error in game {game + 1}: {str(e)}")
        
        finally:
            # Always ensure process is terminated
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # Small delay between games
            time.sleep(1)
    
    return results

def main():
    # Paths
    rl_bot = "kits/rl/main.py"
    do_nothing_bot = "kits/do_nothing/main.py"
    
    print("=" * 60)
    print(f"RL Bot Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test RL bot as player_0
    print("\nTesting RL bot as player_0 (3 games):")
    print("-" * 60)
    p0_results = run_game(rl_bot, do_nothing_bot)
    
    # Test RL bot as player_1
    print("\nTesting RL bot as player_1 (3 games):")
    print("-" * 60)
    p1_results = run_game(do_nothing_bot, rl_bot)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\nRL Bot as Player 0:")
    for result in p0_results:
        print(f"Game {result['game']}: {result['rewards']}")
    
    print("\nRL Bot as Player 1:")
    for result in p1_results:
        print(f"Game {result['game']}: {result['rewards']}")

if __name__ == "__main__":
    main()
