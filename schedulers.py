import math

import beartype


@beartype.beartype
class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)


@beartype.beartype
class Linear(Scheduler):
    def __init__(self, init: float, final: float, n_steps: int):
        self.init = init
        self.final = final
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        return self.init + (self.final - self.init) * (self._step / self.n_steps)


class CosineWarmup(Scheduler):
    def __init__(
        self, init: float, max: float, final: float, n_warmup_steps: int, n_steps: int
    ):
        self.init = init
        self.max = max
        self.final = final
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_warmup_steps:
            # Linear warmup
            return self.init + (self.max - self.init) * (
                self._step / self.n_warmup_steps
            )

        # Cosine decay.
        return self.final + 0.5 * (self.max - self.final) * (
            1
            + math.cos(
                (self._step - self.n_warmup_steps)
                / (self.n_steps - self.n_warmup_steps)
                * math.pi
            )
        )


def _plot_example_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    n_steps = 1000
    xs = np.arange(n_steps)

    schedule = Linear(0.1, 0.9, n_steps)
    ys = [schedule.step() for _ in xs]

    ax1.plot(xs, ys)
    ax1.set_title("Linear")

    schedule = CosineWarmup(0.1, 1.0, 0.0, 200, n_steps)
    ys = [schedule.step() for _ in xs]

    ax2.plot(xs, ys)
    ax2.set_title("CosineWarmup")

    fig.tight_layout()
    fig.savefig("schedules.png")


if __name__ == "__main__":
    _plot_example_schedules()
