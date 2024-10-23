# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Union

from .sequential_mutable_channel import SquentialMutableChannel


class OneShotMutableChannel(SquentialMutableChannel):
    """OneShotMutableChannel is a subclass of SquentialMutableChannel. The
    difference is that a OneShotMutableChannel limits the candidates of the
    choice.

    Args:
        num_channels (int): number of channels.
        candidate_choices (List[Union[float, int]], optional):  A list of
            candidate width ratios. Each candidate indicates how many
            channels to be reserved. Defaults to [].
        choice_mode (str, optional): Mode of choices. Defaults to 'number'.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[float, int]] = [],
                 choice_mode='number',
                 **kwargs):
        super().__init__(num_channels, choice_mode, **kwargs)
        candidate_choices.sort()
        self.candidate_choices = candidate_choices
        if candidate_choices == []:
            candidate_choices.append(num_channels if self.is_num_mode else 1.0)

    @property
    def current_choice(self) -> Union[int, float]:
        """Get current choice."""
        return super().current_choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """Set current choice."""
        assert choice in self.candidate_choices
        SquentialMutableChannel.current_choice.fset(  # type: ignore
            self,  # type: ignore
            choice)  # type: ignore

    def sample_choice(self) -> Union[int, float]:
        """Sample a valid choice."""
        rand_idx = random.randint(0, len(self.candidate_choices) - 1)
        return self.candidate_choices[rand_idx]

    @property
    def min_choice(self) -> Union[int, float]:
        """Get Minimal choice."""
        return self.candidate_choices[0]

    @property
    def max_choice(self) -> Union[int, float]:
        """Get Maximal choice."""
        return self.candidate_choices[-1]

    @property
    def choices(self):
        return self.candidate_choices
