"""Explicit upper-command transport delay; no state or base-command delay."""
from collections import deque
import numpy as np


class CommandDelay:
    def __init__(self, seconds, dt):
        assert seconds >= 0 and dt > 0
        self.steps = round(seconds/dt)
        assert abs(self.steps*dt-seconds) < 1e-10, 'Delay must be an integer number of physics steps.'
        self.queue = deque()

    def apply(self, command):
        if not self.queue:
            self.queue.extend(command.copy() for _ in range(self.steps))
        self.queue.append(command.copy())
        return self.queue.popleft()


def check_delay():
    cases = []
    for steps in [0, 1, 20, 50]:
        delay = CommandDelay(steps*.001, .001)
        commands = np.arange(200*3, dtype=float).reshape(200, 3)
        output = np.array([delay.apply(c) for c in commands])
        expected = commands[np.maximum(0, np.arange(200)-steps)]
        assert np.array_equal(output, expected)
        cases.append({'delay_steps': steps, 'exact_sequence_passed': True})
    return cases
