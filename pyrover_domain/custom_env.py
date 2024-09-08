from pyrover_domain.librovers import rovers, thyme
import numpy as np
import cppyy
import random
from pyrover_domain.custom_pois import DecayPOI, OrderedPOI
from pyrover_domain.custom_sensors import CustomLidar, CustomCamera


# First we're going to create a simple rover
def createRover(obs_radius: float, img_size: float, reward_type: str, sensor_type: str, resolution: float):
    discrete = thyme.spaces.Discrete

    match (reward_type):
        case "global":
            reward = rovers.rewards.Global
        case "difference":
            reward = rovers.rewards.Difference

    match (sensor_type):
        case "lidar":
            sensor = CustomLidar(resolution=resolution, composition_policy=rovers.Density())
            rover = rovers.Rover[CustomLidar, discrete, reward](obs_radius, sensor, reward())
        case "camera":
            sensor = CustomCamera(img_size=img_size)
            rover = rovers.Rover[CustomCamera, discrete, reward](obs_radius, sensor, reward())

    rover.type = "rover"
    return rover


def createDecayPOI(value, obs_rad, coupling, lifespan):
    poi = DecayPOI(value, obs_rad, rovers.CountConstraint(coupling), lifespan)
    return poi


def createOrderedPOI(value, obs_rad, coupling, lifespan, order):
    poi = OrderedPOI(value, obs_rad, rovers.CountConstraint(coupling), lifespan, order)
    return poi


def createStaticPOI(value, obs_rad, coupling):
    countConstraint = rovers.CountConstraint(coupling)
    poi = rovers.POI[rovers.CountConstraint](value, obs_rad, countConstraint)
    return poi


def resolvePositionSpawnRule(position_dict):

    match (position_dict["spawn_rule"]):
        case "fixed":
            return position_dict["fixed"]

        case "random_uniform":
            low_x = position_dict["random_uniform"]["low_x"]
            high_x = position_dict["random_uniform"]["high_x"]
            x = random.uniform(low_x, high_x)
            low_y = position_dict["random_uniform"]["low_y"]
            high_y = position_dict["random_uniform"]["high_y"]
            y = random.uniform(low_y, high_y)
            return [x, y]

        case "random_circle":
            theta = random.uniform(0, 2 * np.pi)
            r = position_dict["random_circle"]["radius"]
            center = position_dict["random_circle"]["center"]
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            return [x, y]


# Let's have a function that builds out the environment
def createEnv(config):
    Env = rovers.Environment[rovers.CustomInit]

    # Aggregate all of the positions of agents
    agent_positions = []
    for rover in config["env"]["rovers"][: config["teaming"]["team_size"]]:
        position = resolvePositionSpawnRule(rover["position"])
        agent_positions.append(position)

    # Aggregate all of the positions of pois
    poi_positions = []
    for poi in config["env"]["pois"]:
        position = resolvePositionSpawnRule(poi["position"])
        poi_positions.append(position)

    rovers_ = [
        createRover(
            obs_radius=rover["observation_radius"],
            img_size=config["env"]["img_sensor_size"],
            reward_type=rover["reward_type"],
            sensor_type=rover["sensor_type"],
            resolution=rover["resolution"],
        )
        for rover in config["env"]["rovers"][: config["teaming"]["team_size"]]
    ]

    agents = rovers_

    rover_pois = []
    for poi in config["env"]["pois"]:

        match (poi["type"]):
            case "static":
                rover_pois.append(
                    createStaticPOI(
                        value=poi["value"],
                        obs_rad=poi["observation_radius"],
                        coupling=poi["coupling"],
                    )
                )

            case "decay":
                rover_pois.append(
                    createDecayPOI(
                        value=poi["value"],
                        obs_rad=poi["observation_radius"],
                        coupling=poi["coupling"],
                        lifespan=poi["lifespan"] * config["ccea"]["num_steps"],
                    )
                )

            case "ordered":
                rover_pois.append(
                    createOrderedPOI(
                        value=poi["value"],
                        obs_rad=poi["observation_radius"],
                        coupling=poi["coupling"],
                        order=poi["order"],
                        lifespan=poi["lifespan"] * config["ccea"]["num_steps"],
                    )
                )

    pois = rover_pois

    env = Env(
        rovers.CustomInit(agent_positions, poi_positions),
        agents,
        pois,
        width=cppyy.gbl.ulong(config["env"]["map_size"][0]),
        height=cppyy.gbl.ulong(config["env"]["map_size"][1]),
    )
    return env


# Alright let's give this a try
def main():
    config = {
        "env": {
            "agents": {
                "rovers": [
                    {
                        "observation_radius": 3.0,
                        "reward_type": "Global",
                        "resolution": 90,
                        "position": {"spawn_rule": "fixed", "fixed": [10.0, 10.0]},
                    }
                ],
            },
            "pois": {
                "rover_pois": [
                    {
                        "value": 1.0,
                        "observation_radius": 1.0,
                        "coupling": 1,
                        "position": {"spawn_rule": "fixed", "fixed": [40.0, 40.0]},
                    }
                ],
            },
            "map_size": [50.0, 50.0],
        }
    }

    env = createEnv(config)

    states, rewards = env.reset()

    print("States:")
    for ind, state in enumerate(states):
        print("agent " + str(ind))
        print(state.transpose())

    print("Rewards:")
    for ind, reward in enumerate(rewards):
        print("reward " + str(ind))
        print(reward)


if __name__ == "__main__":
    main()
