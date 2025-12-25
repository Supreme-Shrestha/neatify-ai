import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.population import Population

def dummy_fitness_with_persistence(genomes):
    # Assign random fitness, but keep existing high fitness if any (to simulate learning)
    # Actually, for this test, we want to see if SPECIFIC high fitness values are preserved.
    for g in genomes:
        if g.fitness == 0.0: # Only assign if not already set (elites have fitness copied)
             g.fitness = random.uniform(0, 10)

def test_elitism():
    pop = Population(pop_size=20, num_inputs=2, num_outputs=1)
    pop.config.elitism_count = 3
    
    # Manually set high fitness for 3 genomes
    pop.genomes[0].fitness = 100.0
    pop.genomes[1].fitness = 90.0
    pop.genomes[2].fitness = 80.0
    
    # Run one generation
    # We need a fitness function that doesn't overwrite the elite's fitness if we want to check preservation
    # But run_generation calls fitness_function(self.genomes) at the start.
    # So the fitness function MUST re-evaluate.
    # For the test, let's define a fitness function that assigns fitness based on ID or some marker, 
    # OR we check if the genomes in the new generation are copies of the old ones.
    
    # Better approach:
    # 1. Run generation.
    # 2. Check if the top 3 fitness values from prev gen are present in new gen.
    
    # Assign initial fitness
    for i, g in enumerate(pop.genomes):
        g.fitness = float(i) # 0 to 19
        
    print("Initial best fitnesses:", sorted([g.fitness for g in pop.genomes], reverse=True)[:5])
    
    # Define fitness function that assigns NEW random fitnesses to everyone
    # BUT elites should have been copied BEFORE this? 
    # Wait, run_generation calls fitness_function FIRST.
    # So:
    # 1. fitness_function evaluates current population.
    # 2. Elites are selected based on this fitness.
    # 3. Elites are copied to new_genomes.
    # 4. Next generation starts.
    # 5. Next run_generation calls fitness_function on NEW genomes.
    
    # So to test preservation, we need to check the population AFTER run_generation but BEFORE the next fitness evaluation.
    # But run_generation replaces self.genomes.
    # So we can check pop.genomes immediately after run_generation.
    # The elites in the new population will have the OLD fitness values until re-evaluated.
    
    def null_fitness(genomes):
        # Do nothing, assuming fitness is already set or we set it manually
        pass
        
    # We need to manually set fitnesses because run_generation calls fitness_function
    # Let's wrap the manual setting in the fitness function
    def setup_fitness(genomes):
        for i, g in enumerate(genomes):
            # If it's a new genome (fitness 0 usually), assign random
            # If it's an elite (fitness > 0), it keeps it?
            # Actually, usually we re-evaluate everyone every gen.
            # But for this test, we want to verify the COPY mechanism.
            if g.fitness == 0.0:
                g.fitness = random.uniform(0, 50)
                
    # Let's just set fitnesses manually, then call run_generation with a dummy function
    # But run_generation calls fitness_function first thing.
    
    # Correct flow for test:
    # 1. Initialize pop.
    # 2. Set fitnesses manually.
    # 3. Call run_generation with a no-op fitness function (so it uses our manual values).
    # 4. Check if best values are in the new population.
    
    for i, g in enumerate(pop.genomes):
        g.fitness = 100.0 + i # 100 to 119
        
    best_fitnesses = sorted([g.fitness for g in pop.genomes], reverse=True)[:3]
    print("Top 3 fitnesses before:", best_fitnesses)
    
    pop.run_generation(null_fitness)
    
    current_fitnesses = sorted([g.fitness for g in pop.genomes], reverse=True)[:3]
    print("Top 3 fitnesses after: ", current_fitnesses)
    
    if best_fitnesses == current_fitnesses:
        print("SUCCESS: Elites preserved.")
    else:
        print("FAILURE: Elites lost.")

if __name__ == "__main__":
    test_elitism()
