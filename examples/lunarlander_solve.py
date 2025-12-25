"""
NEATify Benchmark: LunarLander Continuous Control.

This script demonstrates NEAT solving a complex reinforcement learning
task from the Gymnasium library. 

Key Concepts:
1. Environment state mapping (8 inputs) to discrete actions (4 outputs).
2. Fitness evaluation across multiple episodes to reduce noise.
3. Checkpoint persistence: Saving population state every 10 generations.
4. Deployment: Loading the best evolved genome for a rendered demonstration.

Target: Achieve a mean reward of 200+ (solved state).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gymnasium as gym
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig
from neatify.population import Population
from neatify.pytorch_adapter import NeatModule
from neatify.checkpoint import Checkpoint

def evaluate_genome_lunarlander(genome, num_episodes=3, render=False):
    """
    Simulate a genome in the LunarLander environment.
    
    Args:
        genome (Genome): The policy network to evaluate.
        num_episodes (int): Number of trials to average rewards over.
        render (bool): If True, displays the game GUI.
        
    Returns:
        float: Mean reward achieved over num_episodes. Returns -1000.0 on failure.
    """
    try:
        model = NeatModule(genome, use_sparse=False, trainable=False)
        
        # Cross-version support for Gymnasium
        try:
            env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        except Exception:
            try:
                env = gym.make('LunarLander-v2', render_mode='human' if render else None)
            except Exception as e:
                print(f"CRITICAL: LunarLander environment not found. Check box2d installation. {e}")
                return -1000.0
        
        total_reward = 0.0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            truncated = False
            
            while not (done or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs = model(obs_tensor).squeeze()
                
                # Deterministic action selection via argmax
                action = torch.argmax(action_probs).item() if action_probs.dim() > 0 else 0
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        env.close()
        return total_reward / num_episodes
        
    except Exception as e:
        print(f"Error evaluating genome: {e}")
        return -1000.0

def fitness_function(genomes):
    """
    Standard fitness evaluation wrapper for the Population loop.
    
    Assigns results of LunarLander simulation to each genome's fitness attribute.
    """
    for i, genome in enumerate(genomes):
        genome.fitness = evaluate_genome_lunarlander(genome, num_episodes=2)
        if i % 10 == 0:
            print(f"  Processed {i}/{len(genomes)} genomes...")

def main():
    """Main evolution and benchmarking loop."""
    print("=" * 70)
    print("LUNARLANDER SOLVER - NEATify Benchmark")
    print("=" * 70)
    
    # 1. Hyperparameter Settings
    config = EvolutionConfig()
    config.population_size = 50
    config.prob_mutate_weight = 0.8
    config.prob_add_connection = 0.2
    config.prob_add_node = 0.05
    config.elitism_count = 3
    config.weight_mutation_power = 0.5
    
    print("\n[1] Initializing population...")
    print("  Model Topology: 8 inputs -> 4 discrete outputs")
    
    pop = Population(pop_size=50, num_inputs=8, num_outputs=4, config=config)
    
    # 2. Main Eevolutionary Cycle
    print("\n[2] Starting Evolution...")
    print("-" * 70)
    
    best_fitness_ever = float('-inf')
    best_genome_ever = None
    
    for generation in range(100):
        print(f"\nGeneration {generation}")
        
        pop.run_generation(fitness_function)
        
        best = max(pop.genomes, key=lambda g: g.fitness)
        avg_fitness = sum(g.fitness for g in pop.genomes) / len(pop.genomes)
        
        if best.fitness > best_fitness_ever:
            best_fitness_ever = best.fitness
            best_genome_ever = best.copy()
        
        print(f"  Best: {best.fitness:7.2f} | Avg: {avg_fitness:7.2f} | Best Ever: {best_fitness_ever:7.2f}")
        
        # Convergence Check
        if best.fitness >= 200:
            print(f"\n✓ LUNARLANDER SOLVED at generation {generation}!")
            break
        
        # Periodic Checkpointing
        if generation % 10 == 0 and generation > 0:
            Checkpoint.save(pop, f"lunarlander_checkpoint_gen{generation}.pkl")
            print(f"  Snapshot saved to 'lunarlander_checkpoint_gen{generation}.pkl'")
    
    # 3. Save Final Results
    if best_genome_ever:
        Checkpoint.save_best(best_genome_ever, "lunarlander_best.pkl", 
                           {"fitness": best_fitness_ever, "generation": generation})
        print(f"\n✓ Success! Best genome exported to 'lunarlander_best.pkl'")
    
    # 4. Final Verification
    print("\n" + "=" * 70)
    print("[3] Final Visual Verification")
    print("=" * 70)
    
    if best_genome_ever:
        print("Testing champion on 3 high-fidelity episodes...")
        test_reward = evaluate_genome_lunarlander(best_genome_ever, num_episodes=3, render=True)
        print(f"Benchmark Reward: {test_reward:.2f}")
    
    print("\nExecution complete.")

if __name__ == "__main__":
    main()
