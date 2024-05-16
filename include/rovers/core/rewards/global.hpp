#ifndef THYME_ENVIRONMENTS_ROVERS_GLOBAL
#define THYME_ENVIRONMENTS_ROVERS_GLOBAL

#include <rovers/core/detail/pack.hpp>

namespace rovers::rewards {

/*
 *
 * Default environment reward: checks if all constraints are satisfied
 *
 */
class Global {
   public:
    [[nodiscard]] double compute(const AgentPack& pack) const {
        // TODO pass in a view of POIContainer filtered by observed()
        // TODO Keep filtering over this view for speed-up
        double reward = 0.0;
        for (const auto& poi : pack.entities) {
            // if (poi->observed()) continue;
            const auto& c = poi->constraint_satisfied({poi, pack.agents, pack.entities});
            std::cout << "c: " << c << std::endl;
            reward = reward + poi->value()*poi->constraint_satisfied({poi, pack.agents, pack.entities});
        }
        // reset pois
        // for (const auto& poi : pack.entities) poi->set_observed(false);
        return reward;
    }
};

}  // namespace rovers::rewards

#endif