from pyrover_domain.librovers import rovers  # import bindings.
import numpy as np

"""
A Decaying POI that dissapears over time. 
T:
    Total time before complete decay, playing with this value speeds up or slows down decay. Usually multiples of T.
"""


class DecayPOI(rovers.IPOI):
    def __init__(
        self,
        value: float,
        obs_radius: float,
        constraintPolicy: rovers.IConstraint,
        lifespan: int,
        decay_start: float = 0,
        decay_value: bool = True,
    ):
        super().__init__(value, obs_radius)

        # we will use the stealth combined with a constraint defined in the lib
        self.time_step = 0
        self.final_val = 1e-02
        self.constraintPolicy = constraintPolicy
        self.visible = True
        self.init_value = value
        self.decay_rate = -np.log(self.final_val) / lifespan
        self.tolerance = self.final_val * value
        self.decay_start = decay_start
        self.decay_value = decay_value

    # Use a library defined or custom constraint policy with stealth
    def constraint_satisfied(self, entity_pack):

        if not self.visible:
            return False  # "you can't see me"

        return self.constraintPolicy.is_satisfied(entity_pack)

    # tick() is called at every time step (tick) by the library.
    # blink based on a coin toss and stealth mastery
    def tick(self):

        if not self.visible:
            return

        decayed_value = self.init_value * np.power((1 - self.decay_rate), self.time_step - self.decay_start)

        if self.decay_value:
            self.set_value(decayed_value)

        if self.decay_start <= self.time_step:
            if np.isclose(
                decayed_value,
                0,
                rtol=self.tolerance,
                atol=self.tolerance,
            ):
                self.set_value(0)
                self.visible = False

        self.time_step += 1
