"""1D variant of VirtualLab: only build_and_train differs (uses 1D model)."""
from lab import VirtualLab
from genome_1d import ConfigurableNeuralOperator1D
from fitness import FitnessScore, evaluate_model


class VirtualLab1D(VirtualLab):
    def build_and_train(self, train_x, train_y, test_x, test_y):
        if self.current_genome is None:
            return None
        try:
            self.current_model = ConfigurableNeuralOperator1D(self.current_genome)
            n_params = self.current_model.count_parameters()
            print(f"    Lab {self.lab_id}: {n_params:,} params, "
                  f"blocks={self.current_genome.block_sequence}")
            self.current_fitness = evaluate_model(
                self.current_model, train_x, train_y, test_x, test_y, self.config)
            if self.current_fitness.accuracy > self.best_fitness:
                self.best_fitness = self.current_fitness.accuracy
                self.best_genome = self.current_genome.copy()
            return self.current_fitness
        except Exception as e:
            print(f"    Lab {self.lab_id}: BUILD/TRAIN FAILED - {type(e).__name__}: {e}")
            self.current_fitness = FitnessScore()
            self.current_model = None
            return self.current_fitness
