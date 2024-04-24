#ifndef THYME_ENVIRONMENTS_ROVERS_INIT_CUSTOM
#define THYME_ENVIRONMENTS_ROVERS_INIT_CUSTOM

#include <cmath>

namespace rovers {

/*
 *
 * agent/entity initialization policy for custom placement
 *
 */
class CustomInit {
   public:
    CustomInit(std::vector<std::vector<double>> rover_positions = {}, std::vector<std::vector<double>> poi_positions = {}) : m_rover_positions(rover_positions), m_poi_positions(poi_positions) {}

    template <typename RoverContainer, typename POIContainer>
    void initialize(RoverContainer& rovers, POIContainer& pois) {
        initialize_rovers(rovers);
        initialize_poi(pois);
    }

   private:
    // template <std::ranges::range RoverContainer>
    template <typename RoverContainer>
    void initialize_rovers(RoverContainer& rovers) {
        double x, y;
        for (std::size_t i = 0; i < rovers.size(); ++i) {
            x = m_rover_positions[i][0];
            y = m_rover_positions[i][1];
            rovers[i]->set_position(x, y);
        }
    }
    template <typename POIContainer>
    void initialize_poi(POIContainer& pois) {
        double x, y;
        for (std::size_t i = 0; i < pois.size(); ++i) {
            x = m_poi_positions[i][0];
            y = m_poi_positions[i][1];
            pois[i]->set_position(x, y);
        }
    }

   private:
    std::vector<std::vector<double>> m_rover_positions;
    std::vector<std::vector<double>> m_poi_positions;
};
}  // namespace rovers

#endif
