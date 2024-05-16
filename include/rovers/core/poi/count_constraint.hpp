#ifndef THYME_ENVIRONMENTS_ROVERS_POI_COUNT_CONSTRAINT
#define THYME_ENVIRONMENTS_ROVERS_POI_COUNT_CONSTRAINT

#include <rovers/core/poi/poi.hpp>
#include <rovers/core/rover/rover.hpp>
#include <rovers/utilities/math/norms.hpp>

namespace rovers {

/*
 *
 * Constraint satifisted by a count of observations
 *
 */
class CountConstraint {
   public:
    explicit CountConstraint(size_t count = 3) : count_constraint(count) {}

    [[nodiscard]] double is_satisfied(const EntityPack& entity_pack) const {
        bool constraint_satisfied = false;
        size_t count = 0;
        std::vector<double> dists;
        for (const auto& rover : entity_pack.agents) {
            double dist = l2_norm(rover->position(), entity_pack.entity->position());
            dists.push_back(dist);
            if (dist <= rover->obs_radius() && dist <= entity_pack.entity->obs_radius()) {
                ++count;
                if (count >= count_constraint) constraint_satisfied = true;
            }
        }
        if (constraint_satisfied) {
            // Get the n closest rovers

            // Sort distances in ascending order (first values are lowest)
            std::sort(dists.begin(), dists.end());

            // Now make sure no values are less than 1.0
            for (int i=0; i<dists.size(); ++i) {
                dists[i] = std::max(dists[i], 1.0);
            }

            // Now calculate how well this constraint was satisfied
            double constraint_value = count_constraint;
            for (int i=0; i<count_constraint; ++i) {
                constraint_value = constraint_value * 1.0/dists[i];
            }
            return constraint_value;
        }
        return 0.0;
    }

   private:
    size_t count_constraint;
};

}  // namespace rovers

#endif