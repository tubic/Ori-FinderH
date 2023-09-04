from Z_Unit import for_ga
from Model_Unit import MainModel
import random 
import torch 
from Bio         import SeqIO
from matplotlib  import pyplot as plt
torch.set_num_threads(100)


def count_gc(sequence):
    return (sequence.count('G') + sequence.count('C')) / len(sequence)


class GA:
    def __init__(self, group_nums, init_length, fitness_function, output_path, gc_output_path, fitness_output_path) -> None:
        self.group_nums = group_nums
        self.base_list  = ["A", "C", "G", "T"]
        self.init_length = init_length
        self.init_sequence_group = None
        self.fitness_function = fitness_function
        self.output_path = output_path
        self.gc_output_path = gc_output_path
        self.fitness_output_path = fitness_output_path
    
    def init_sequence(self):
        print("Start init sequence group.")
        init_sequence_group = []
        for _ in range(self.group_nums):
            tmp_sequence = "".join(["G" for i in range(self.init_length)])
            init_sequence_group.append(tmp_sequence)
        self.init_sequence_group = init_sequence_group
    
    def base_mutation(self, sequence):
        mutation_position = random.randint(0 , len(sequence)-1)
        mutation_base     = random.choice(self.base_list)
        while mutation_base == sequence[mutation_position]:
            mutation_base = random.choice(self.base_list)
        sequence = list(sequence)
        sequence[mutation_position] = mutation_base
        return "".join(sequence)

    def interval_mutation(self, sequence):
        interval_length = 50
        insert_position = random.randint(0 , len(sequence)-50-1)
        insert_interval = "".join([random.choice(self.base_list) for _ in range(interval_length)])
        sequence = sequence[:insert_position] + insert_interval + sequence[insert_position+50:]
        return sequence

    def insert_mutation(self, sequence):
        insert_position = random.randint(0 , len(sequence)-1)
        insert_base     = random.choice(self.base_list)
        new_sequence    = sequence[:insert_position] + insert_base + sequence[insert_position:]
        return new_sequence
    
    def delet_mutation(self, sequence):
        if len(sequence) >= 200:
            sequence       = list(sequence)
            for _ in range(random.randint(1, 100)):
                delet_position = random.randint(0 , len(sequence)-1)
                del sequence[delet_position]
            return "".join(sequence)
        return sequence
    
    def structure_mutation(self, sequence):
        mutation_position = random.randint(0 , len(sequence)-1)
        sequence          = sequence[mutation_position:] + sequence[:mutation_position]
        return sequence
    
    def get_fitness(self, sequence_list):
        test_tensor = torch.Tensor([for_ga(sequence) for sequence in sequence_list])
        fitness     = [i[1].item() for i in self.fitness_function(test_tensor)]
        return fitness
    
    def variation(self, sequence_list):
        new_sequence_list = []
        for sequence in sequence_list:
            mutation_type = random.choice([1, 2, 3, 4, 5])
            if mutation_type == 1:
                new_sequence_list.append(self.base_mutation(sequence))
            elif mutation_type == 2:
                new_sequence_list.append(self.insert_mutation(sequence))
            elif mutation_type == 3:
                new_sequence_list.append(self.delet_mutation(sequence))
            elif mutation_type == 4:
                new_sequence_list.append(self.structure_mutation(sequence))
            elif mutation_type == 5:
                new_sequence_list.append(self.interval_mutation(sequence))
        return new_sequence_list

    def select(self, population, fitness_values, selection_size, tournament_size):
        new_population, new_fitness_values = [], []
        mean_score = sum(fitness_values) / len(fitness_values)
        for seq, seq_score in zip(population, fitness_values):
            if seq_score >= mean_score:
                new_population.append(seq)
                new_fitness_values.append(seq_score)
        population, fitness_values = new_population, new_fitness_values
        selected_indices = []
        for _ in range(selection_size):
            tournament_candidates = random.sample(range(len(population)), tournament_size)
            tournament_winner = max(tournament_candidates, key=lambda index: fitness_values[index])
            selected_indices.append(tournament_winner)
        selected_population = [population[i] for i in selected_indices]
        return selected_population

    def start(self):
        self.init_sequence()
        self.init_sequence_group = self.variation(self.init_sequence_group)
        self.fitness_list        = self.get_fitness(self.init_sequence_group)
        gc_output_path           = open(self.gc_output_path, "w", encoding="UTF-8")
        fitness_output_path      = open(self.fitness_output_path, "w", encoding="UTF-8")
        gc_list                  = [count_gc(seq) for seq in self.init_sequence_group] 
        while min(self.fitness_list) <= 0.5:
            print(f"Fitness-Max:{max(self.fitness_list)}, Mean:{sum(self.fitness_list) / len(self.fitness_list)}, Max:{min(self.fitness_list)}")
            print(f"GC-MAX:{max(gc_list)}, Mean:{sum(gc_list) / len(gc_list)}, Min:{min(gc_list)}")
            gc_output_path.write(f"GC-MAX:{max(gc_list)}, Mean:{sum(gc_list) / len(gc_list)}, Min:{min(gc_list)}\n")
            fitness_output_path.write(f"Fitness-Max:{max(self.fitness_list)}, Mean:{sum(self.fitness_list) / len(self.fitness_list)}, Max:{min(self.fitness_list)}\n")
            self.init_sequence_group = self.select(self.init_sequence_group, self.fitness_list, selection_size=self.group_nums, tournament_size=3)
            self.init_sequence_group = self.variation(self.init_sequence_group)
            self.fitness_list        = self.get_fitness(self.init_sequence_group)
            gc_list                  = [count_gc(seq) for seq in self.init_sequence_group] 
        print(f"Fitness-Max:{max(self.fitness_list)}, Mean:{sum(self.fitness_list) / len(self.fitness_list)}, Min:{min(self.fitness_list)}")
        print(f"GC-MAX:{max(gc_list)}, Mean:{sum(gc_list) / len(gc_list)}, Min:{min(gc_list)}")
        gc_output_path.write(f"GC-MAX:{max(gc_list)}, Mean:{sum(gc_list) / len(gc_list)}, Min:{min(gc_list)}\n")
        fitness_output_path.write(f"Fitness-Max:{max(self.fitness_list)}, Mean:{sum(self.fitness_list) / len(self.fitness_list)}, Max:{min(self.fitness_list)}\n")
        output_file = open(f"{self.output_path}", "w", encoding="UTF-8")
        for sequence_id, sequence in enumerate(self.init_sequence_group):
            output_file.write(f">{sequence_id+1}\n{sequence}\n")
        output_file.close()
        gc_output_path.close()
        fitness_output_path.close()


def load_sequence_list(fasta_path):
    return [str(seq.seq).upper() for seq in SeqIO.parse(fasta_path, 'fasta')]

        
def random_selection_sequence(sequence_list, output_path):
    random_sequence = random.choices(sequence_list, k=1000)
    new_file        = open(output_path, "w", encoding="UTF-8")
    for seq in random_sequence:
        new_file.write(f">{seq.id}\n{seq.seq}\n")
    new_file.close()


def main(gen_time):
    group_nums = 1000
    init_length = 1000
    fitness_function = torch.load("ori_finder_h.pkl", map_location="cpu")
    output_path = f"Gen/gen_{gen_time}.fasta"
    gc_output_path = f"Log/gc_{gen_time}.log"
    fitness_output_path = f"Log/fitness_{gen_time}.log"
    ga = GA(group_nums, init_length, fitness_function, output_path, gc_output_path, fitness_output_path)
    ga.start()


if __name__ == "__main__":
    main(12)