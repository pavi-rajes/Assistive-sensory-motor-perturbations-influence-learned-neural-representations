/**
 * Target.cpp | bmi_model
 */

#include "Target.hpp"
#include <stdexcept>
#include <iostream>

Target::Target(int dim, int nb_targets, double distance_to_target):
dim(dim), nb_targets(nb_targets), distance_to_target(distance_to_target) {
    
    targets.reserve(nb_targets);
    
    for (int i(0);i<nb_targets;i++){
        if (dim == 2) {
            Vector t(dim);
            t << distance_to_target * cos(2 * M_PI * i / double(nb_targets)), distance_to_target * sin(2 * M_PI * i / double(nb_targets));
            targets.push_back(t);
        } else {
            throw std::out_of_range("Targets not yet defined in dimension other than 2. Exiting");
        }
    }    
}


Vector Target::operator()(int i){
    return targets[i];
}

Matrix Target::get_all_targets_as_matrix(){
    Matrix tgts(get_nb_targets(), dim);
    for (int i=0; i<get_nb_targets(); i++){
        tgts.row(i) = targets[i].transpose();
    }
    return tgts;
}

Vector get_target_direction_from_id(int target_id, int nb_targets, int dim){
    if (dim != 2) {
        std::cerr << "In function get_target_direction_from_id. " << "Dim = " << dim << ": dimensions other than 2 are not supported\n";
        std::exit(1);
    } else {
        Vector target_direction(dim);
        target_direction << cos(2 * M_PI * target_id / double(nb_targets)), sin(2 * M_PI * target_id / double(nb_targets));
        return target_direction;
    }
}
