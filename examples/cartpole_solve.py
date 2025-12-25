import gymnasium as gym
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import neatify
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify import Population, NeatModule

def cartpole_fitness(genomes):
    # Create environment
    # gym might not be installed, so we need to handle that or assume it is.
    # For this script, we'll try to import gym.
    try:
        env = gym.make("CartPole-v1")
    except:
        print("Gymnasium not installed or CartPole not found.")
        return

    for genome in genomes:
        model = NeatModule(genome)
        observation, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Prepare input
            input_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # Batch size 1
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
                
            # Action selection (0 or 1)
            action = 1 if output.item() > 0.5 else 0
            
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if total_reward >= 200: # Cap at 200 for speed
                break
                
        genome.fitness = total_reward
        env.close()

def main():
    # CartPole-v1 has 4 inputs, 2 outputs (but we can use 1 output with threshold)
    # Let's use 1 output: < 0.5 -> Left, >= 0.5 -> Right
    pop = Population(pop_size=50, num_inputs=4, num_outputs=1)
    
    generations = 20
    for gen in range(generations):
        pop.run_generation(cartpole_fitness)
        
        best_genome = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Generation {gen}: Best Fitness = {best_genome.fitness:.1f}")
        
        if best_genome.fitness >= 199:
            print("Solved!")
            break
            
    print("Done.")

if __name__ == "__main__":
    main()
