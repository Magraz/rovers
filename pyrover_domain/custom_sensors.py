from pyrover_domain.librovers import rovers  # import bindings.
import numpy as np
import logging 
import cppyy
 
# Creating an object
logger = logging.getLogger()

"""
A custom Lidar
"""
class CustomLidar(rovers.Lidar[rovers.Density]):
    def __init__(self, resolution, composition_policy):

        super().__init__(resolution, composition_policy)

        self.m_resolution = resolution
        self.m_composition = composition_policy
    
    def calculateDistance(self, position_0, position_1):
        return np.linalg.norm([position_0.x-position_1.x, position_0.y-position_1.y])

    def calculateAngle(self, position_0, position_1):
        pos0 = np.array([position_0.x, position_0.y])
        pos1 = np.array([position_1.x, position_1.y])

        # Create a vector from position 0 to 1
        vec = pos1 - pos0
        
        # Take the arctan2 of the y, x of that vector
        angle = np.arctan2(vec[1], vec[0]) * 180./np.pi
        if angle < 0:
            angle += 360.
            
        return angle

    def scan(self, agent_pack):
        num_sectors = int(360. / self.m_resolution)
        poi_values = [[] for _ in range(num_sectors)]
        rover_values = [[] for _ in range(num_sectors)]
        agent = agent_pack.agents[agent_pack.agent_index]

        # Observe POIs
        for sensed_poi in agent_pack.entities:

            # Get angle and distance to POI
            angle = self.calculateAngle(agent.position(), sensed_poi.position())
            distance = self.calculateDistance(agent.position(), sensed_poi.position())
            
            logger.debug(f"angle: {angle}")
            logger.debug(f"distance: {distance}")

            # Skip this POI if it is out of range
            if distance > agent.obs_radius():
                logger.debug("continue, out of range")
                continue
            
            # Bin the POI according to where it was sensed
            if angle < 360.0:
                sector = int( angle / self.m_resolution )
            else:
                sector = 0

            # print("sector: ", sector, type(sector))
            poi_values[sector].append(sensed_poi.value() / max([0.001, distance**2]))

        logger.debug("Observe Agents")

        # Observe agents
        for i in range(agent_pack.agents.size()):

            logger.debug("Sensing agent")

            # Do not observe yourself
            if i == agent_pack.agent_index:
                logger.debug("Nope, that one is me")
                continue

            # Get angle and distance to sensed agent
            sensed_agent = agent_pack.agents[i]
            angle = self.calculateAngle(agent.position(), sensed_agent.position())
            distance = self.calculateDistance(agent.position(), sensed_agent.position())

            logger.debug(f"angle: {angle}")
            logger.debug(f"distance: {angle}")

            # Skip the agent if the sensed agent is out of range
            if distance > agent.obs_radius():
                # print("continue, out of range")
                continue

            # Get the bin for this agent
            if angle < 360.0:
                sector = int( angle / self.m_resolution )
            else:
                sector = 0

            rover_values[sector].append(1.0 / max([0.001, distance**2]))

        logger.debug(f"rover_values: {rover_values}")
        logger.debug(f"poi_values: {poi_values}")

        # Encode the state
        logger.debug("Encoding state")

        # num_sectors*2 since we have 4 sectors and 2 types of distances, rovers and pois, these are accumulated into a single value by the lidar.hpp Density class
        state = np.array([-1.0 for _ in range(num_sectors*2)])

        logger.debug(f"state: {state}")

        for i in range(num_sectors):
            logger.debug(f"Building sector {i}")

            num_rovers = len(rover_values[i])
            num_pois = len(poi_values[i])

            if num_rovers > 0:

                logger.debug("num_rovers > 0")
                logger.debug(f"rover_values[{str(i)}]: {rover_values[i]} {type(rover_values[i])} {type(rover_values[i][0])}")
                logger.debug(f"num_rovers: {type(num_rovers)}")

                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()

                for r in rover_values[i]:
                    cpp_vector.push_back(r)

                state[i] = self.m_composition.compose(cpp_vector, 0.0, num_rovers)
            
            if num_pois > 0:
                logger.debug("num_pois > 0")
                logger.debug(f"poi_values[{str(i)}]: {poi_values[i]} {type(poi_values[i])} {type(poi_values[i][0])}")
                logger.debug(f"num_pois: {type(num_pois)}")

                # Convert poi_values[i] to a std::vector<double> to satisfy cppyy
                # Not sure why this is necessary sometimes and other times not necessary
                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()

                for p in poi_values[i]:
                    cpp_vector.push_back(p)

                state[num_sectors + i] = self.m_composition.compose(cpp_vector, 0.0, num_pois)

        return rovers.tensor(state)