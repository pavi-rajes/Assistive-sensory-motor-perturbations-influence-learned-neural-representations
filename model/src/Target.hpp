/**
 * Target.hpp | bmi_model
 *
 * Description:
 * -----------
 * Defines a simple object containing the fixed targets.
 */

//  Target.hpp | rnn4bci_cpp

#ifndef Target_hpp
#define Target_hpp

#include <vector>
#include <Eigen/Dense>
#include "globals.hpp"

class Target {
    int dim;                   // number of dimensions that specify a target's position (typically dim=2, i.e. targets in a plane)
    int nb_targets;            // number of targets
    double distance_to_target; // radial distance from origin to target, in meters.
    std::vector<Vector> targets;
    
public:
    Target(int dim, int nb_targets, double distance_to_target);
    Vector operator()(int i);
    std::vector<Vector> get_all_targets() {return targets;}
    Matrix get_all_targets_as_matrix();

    int get_nb_targets() {return nb_targets;}
    double get_distance_to_target() {return distance_to_target;}
};

Vector get_target_direction_from_id(int target_id, int nb_targets, int dim);

#endif /* Target_hpp */
