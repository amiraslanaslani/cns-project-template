import torch

from cnsproject.mechanism.mechanism import AbstractMechanism, AFTER, BEFORE
from cnsproject.network.neural_populations import PopulationVariables, NeuralPopulation


class WinnerTakeAllInhibition(AbstractMechanism):
    population_object: NeuralPopulation
    invalid_choices: torch.Tensor
    winners: torch.Tensor
    found_ks: int

    def __init__(
            self,
            k: int,
            lateral_inhibitions_radius: int,
            filters_dim: int = -3,
            inactivity_counter: int = 1
    ):
        super().__init__()
        self.k = k
        self.lateral_inhibitions_radius = lateral_inhibitions_radius
        self.filters_dim = filters_dim
        self.inactivity_counter = inactivity_counter

    def main(self):
        self.register("set_spikes", AFTER, self.after_compute_spike)

    def init(self, population_object):
        shape = population_object.shape
        self.population_object = population_object
        self.found_ks = 0
        self.invalid_choices = torch.zeros(shape)

    def reset(self):
        self.init(self.population_object)

    def forward(self):
        invalid_choices = self.invalid_choices
        self.init(self.population_object)
        self.invalid_choices = invalid_choices
        self.invalid_choices[invalid_choices >= 1] = self.invalid_choices[invalid_choices >= 1] - 1

    def after_compute_spike(self, population_object, returned_value):
        winners = torch.zeros(population_object.shape)
        potentials = getattr(population_object, PopulationVariables.RB_POTENTIAL)
        winner = self.find_winner(potentials, population_object.threshold)
        while self.found_ks < self.k and winner is not None:
            winners[tuple(winner)] = 1
            self.found_ks = self.found_ks + 1

            # Remove winner's filter from valid choices
            winner_filter_valid_choices = self.invalid_choices.select(self.filters_dim, winner[self.filters_dim])
            winner_filter_valid_choices[:] = self.inactivity_counter

            # Remove laterally inhibited neurons from valid choices
            winner_h = winner[-2]
            winner_w = winner[-1]
            r = self.lateral_inhibitions_radius
            self.invalid_choices[
                ...,
                max(0, winner_h - r):min(population_object.shape[-2], winner_h + r + 1),
                max(0, winner_w - r):min(population_object.shape[-1], winner_w + r + 1)
            ] = self.inactivity_counter
            winner = self.find_winner(potentials, population_object.threshold)
        setattr(population_object, PopulationVariables.RB_SPIKES, winners.bool())

    def find_winner(self, potentials: torch.Tensor, threshold):
        valids = potentials.clone().detach()
        valids[self.invalid_choices.bool()] = threshold.float() - 1

        max_valid = valids[valids != 0].max()
        if max_valid > threshold:
            winner = (valids == max_valid).nonzero()[0]
            return winner
        return None
