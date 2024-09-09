from pyrover_domain.librovers import rovers  # import bindings.
import numpy as np

"""
A Decaying POI that dissapears over time. 
"""


class DecayPOI(rovers.IPOI):
    def __init__(
        self,
        value: float,
        obs_radius: float,
        constraintPolicy: rovers.IConstraint,
        lifespan: int,
        decay_start: float = 0,
        decay_value: bool = False,
        decay_type: str = "exp",
    ):
        super().__init__(value, obs_radius)

        self.time_step = 0
        self.final_val = 1e-02
        self.constraintPolicy = constraintPolicy
        self.visible = True
        self.init_value = value
        self.decay_start = decay_start
        self.decay_value = decay_value
        self.decay_type = decay_type  # exp/linear

        # Set decay type
        match (self.decay_type):
            case "exp":
                self.decay_rate = np.log(self.final_val / self.init_value) / lifespan

    def constraint_satisfied(self, entity_pack):

        if not self.visible:
            return False

        return self.constraintPolicy.is_satisfied(entity_pack)

    def tick(self):

        if not self.visible:
            return

        decayed_value = self.init_value * np.exp(self.decay_rate * (self.time_step - self.decay_start))

        if self.decay_value:
            self.set_value(decayed_value)

        if (self.decay_start <= self.time_step) and (decayed_value < self.final_val):
            self.set_value(0)
            self.visible = False

        self.time_step += 1


class BlinkPOI(rovers.IPOI):
    def __init__(
        self,
        value: float,
        obs_radius: float,
        constraintPolicy: rovers.IConstraint,
        blink_prob: float,
    ):
        super().__init__(value, obs_radius)

        self.time_step = 0
        self.constraintPolicy = constraintPolicy
        self.visible = True
        self.blink_prob = blink_prob

    def constraint_satisfied(self, entity_pack):

        if not self.visible:
            return False

        return self.constraintPolicy.is_satisfied(entity_pack)

    def tick(self):
        self.visible = np.random.choice([True, False], 1, p=[self.blink_prob, 1 - self.blink_prob])[0]


"""
A Ordered POI that can only be observed after all the previous ones have been observed. 
"""


class OrderedPOI(rovers.IPOI):
    def __init__(
        self,
        value: float,
        obs_radius: float,
        constraintPolicy: rovers.IConstraint,
        lifespan: int,
        group: int,
        order: int,
    ):
        super().__init__(value, obs_radius)

        self.time_step = 0
        self.constraintPolicy = constraintPolicy
        self.order = order
        self.group = group
        self.lifespan = lifespan
        self.visible = True
        self.start_decay = False
        self.decay_start_time = 0

    def constraint_satisfied(self, entity_pack):

        if not self.visible:
            return False

        if entity_pack.entity.observed() and not self.start_decay:
            self.start_decay = True
            self.decay_start_time = self.time_step

        for poi in entity_pack.entities:
            if poi.group == self.group:
                if poi.order < self.order:
                    if not poi.observed():
                        return False

        return self.constraintPolicy.is_satisfied(entity_pack)

    def tick(self):

        if not self.visible:
            return

        if self.start_decay and ((self.time_step - self.decay_start_time) >= self.lifespan):
            self.set_value(0)
            self.visible = False

        self.time_step += 1
