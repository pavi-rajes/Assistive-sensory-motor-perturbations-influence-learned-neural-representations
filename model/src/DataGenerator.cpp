/**
 * DataGenerator.cpp | bmi_model
 */

#include "DataGenerator.hpp"
#include <cmath>
#include <vector>
#include <assert.h>
#include "utilities.hpp"

//~~~~~~~~~~~~~~~~~~//
// TARGET GENERATOR //
//~~~~~~~~~~~~~~~~~~//
TargetGenerator::TargetGenerator(int dim, double r) : dim(dim), max_radius(r){
    datatype = DataGenerator::target_only;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> TargetGenerator::generate(int n){
    assert(dim == 2);
    Eigen::MatrixXd X(n, dim+2);
    Eigen::MatrixXd O(n, dim);
    std::uniform_real_distribution<double> unif_radius(0., max_radius);
    std::uniform_real_distribution<double> unif_angle(0., 2*M_PI);
    std::bernoulli_distribution bern;
    std::vector<double> distances_from_center(n);
    std::vector<double> angles(n);
    
    for (int i=0;i<n;++i){
        distances_from_center[i] = unif_radius(rng);
        angles[i] = unif_angle(rng);
    }
    for (int i=0;i<n;++i) X(i, 0) = distances_from_center[i] * cos(angles[i]);
    for (int i=0;i<n;++i) X(i, 1) = distances_from_center[i] * sin(angles[i]);
    for (int i=0;i<n;++i) {
        bool is_manual = bern(rng);
        X(i, 2) = double(is_manual);
        X(i, 3) = double(!is_manual);
    }
    
    for (int i=0;i<n;++i) O(i, 0) = distances_from_center[i] * cos(angles[i]);
    for (int i=0;i<n;++i) O(i, 1) = distances_from_center[i] * sin(angles[i]);
    
    
    return std::make_pair(X.transpose(), O.transpose());
}

std::vector<int> TargetGenerator::find_context(const Eigen::MatrixXd &X, short int context){
    std::vector<int> indices;
    for (int c=0;c<X.cols();c++) { // loop over examples
        if (abs(X(2, c) - double(context)) < 1e-8) indices.push_back(c);
    }
    return indices;
}

int TargetGenerator::get_input_dim(){
    assert(dim == 2);
    return dim + 2;
}

int TargetGenerator::get_output_dim(){
    assert(dim == 2);
    return dim;
}


//~~~~~~~~~~~~~~~~~//
// MIXED GENERATOR //
//~~~~~~~~~~~~~~~~~//
MixedGenerator::MixedGenerator(int dim, double r) : dim(dim), max_radius(r){
    datatype = DataGenerator::target_and_position;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MixedGenerator::generate(int n){
    assert(dim == 2);
    Eigen::MatrixXd X(n, 2*dim + 2);    // Input data : 2*dim for the hand/cursor and target position; +2 for the context
    Eigen::MatrixXd O(n, 3*dim);        // Output data : 2*dim for the hand/cursor and target position; + dim for the relative position;
    
    std::uniform_real_distribution<double> unif_radius(0., max_radius);
    std::uniform_real_distribution<double> unif_angle(0., 2*M_PI);
    std::bernoulli_distribution bern;
    
    std::vector<double> target_distances_from_center(n);
    std::vector<double> target_angles(n);
    std::vector<double> effector_distances_from_center(n);
    std::vector<double> effector_angles(n);
    
    for (int i=0;i<n;++i){
        target_distances_from_center[i] = unif_radius(rng);
        target_angles[i] = unif_angle(rng);
    }
    for (int i=0;i<n;++i){
        effector_distances_from_center[i] = unif_radius(rng);
        effector_angles[i] = unif_angle(rng);
    }
    
    // Construct inputs
    for (int i=0;i<n;++i) X(i, 0) = effector_distances_from_center[i] * cos(effector_angles[i]);
    for (int i=0;i<n;++i) X(i, 1) = effector_distances_from_center[i] * sin(effector_angles[i]);
    for (int i=0;i<n;++i) X(i, 2) = target_distances_from_center[i] * cos(target_angles[i]);
    for (int i=0;i<n;++i) X(i, 3) = target_distances_from_center[i] * sin(target_angles[i]);
    
    for (int i=0;i<n;++i) {
        bool is_manual = bern(rng);
        X(i, 4) = double(is_manual);
        X(i, 5) = double(!is_manual);
    }
    
    // Construct outputs
    for (int col=0; col<2*dim; col++) O.col(col) = X.col(col);
    for (int d=0; d<dim; d++) O.col(2*dim+d) = X.col(d + dim) - X.col(d);
    
    return std::make_pair(X.transpose(), O.transpose());
}


int MixedGenerator::get_input_dim(){
    assert(dim == 2);
    return 2*dim + 2;
}

int MixedGenerator::get_output_dim(){
    assert(dim == 2);
    return 3*dim;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// TORQUE-BASED ARM GENERATOR //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
TorqueBasedArmGenerator::TorqueBasedArmGenerator(int dim, double max_radius, double distance_to_target, const TorqueBasedArm& arm) : dim(dim), distance_to_target(distance_to_target), max_radius(max_radius), arm(arm){
    datatype = DataGenerator::torque_based_arm;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> TorqueBasedArmGenerator::generate(int n){
    assert(dim == 2);
    Eigen::MatrixXd X(n, 5*dim + 2);    // Input data  : 2*dim for the hand/cursor and target position + 3*dim for angular state vector; +2 for the context
    Eigen::MatrixXd O(n, 4*dim);        // Output data : 3*dim for angular state vector; + dim for the relative position;
    
    std::uniform_real_distribution<double> unif(-1., 1.);
    std::uniform_real_distribution<double> unif_radius(0., max_radius/distance_to_target);
    std::uniform_real_distribution<double> unif_target_angle(0., 2*M_PI);
    std::uniform_real_distribution<double> unif_arm_angle(0., 1.); // angle is divided by pi in DualInput
    std::bernoulli_distribution bern;
    
    std::vector<double> shoulder_angle(n);
    std::vector<double> elbow_angle(n);
    std::vector<double> target_distances_from_center(n);
    std::vector<double> target_angles(n);
    
    for (int i=0;i<n;++i){
        target_distances_from_center[i] = unif_radius(rng);
        target_angles[i] = unif_target_angle(rng);
    }
    for (int i=0;i<n;++i){
        shoulder_angle[i] = unif_arm_angle(rng);
        elbow_angle[i] = unif_arm_angle(rng);
    }
    
    // Construct inputs
    Eigen::Vector2d center = arm.get_center_position();
    
    for (int i=0;i<n;++i) {
        double sa = shoulder_angle[i] * M_PI;
        double ea = elbow_angle[i] * M_PI;
        Eigen::Vector2d p = transform_joint_angles_to_end_effector_position(sa, ea, arm.get_l1(), arm.get_l2());
        X(i, 0) = (p(0) - center(0))/distance_to_target;
        X(i, 1) = (p(1) - center(1))/distance_to_target;
    }
    for (int i=0;i<n;++i) X(i, 2) = unif(rng); // shoulder angle
    for (int i=0;i<n;++i) X(i, 3) = unif_arm_angle(rng); // elbow angle
    for (int i=0;i<n;++i) X(i, 4) = unif(rng); // shoulder velocity
    for (int i=0;i<n;++i) X(i, 5) = unif(rng); // elbow velocity
    for (int i=0;i<n;++i) X(i, 6) = unif(rng); // shoulder torque
    for (int i=0;i<n;++i) X(i, 7) = unif(rng); // elbow torque
    for (int i=0;i<n;++i) X(i, 8) = target_distances_from_center[i] * cos(target_angles[i]);
    for (int i=0;i<n;++i) X(i, 9) = target_distances_from_center[i] * sin(target_angles[i]);
    
    for (int i=0;i<n;++i) {
        bool is_manual = bern(rng);
        X(i, 10) = double(is_manual);
        X(i, 11) = double(!is_manual);
    }
    
    // Construct outputs
    for (int col=0; col<3*dim; col++) O.col(col) = X.col(col+dim);
    for (int d=0; d<dim; d++) O.col(3*dim+d) = X.col(4*dim+d) - X.col(d);
    
    return std::make_pair(X.transpose(), O.transpose());
}

int TorqueBasedArmGenerator::get_input_dim(){
    assert(dim == 2);
    return 5*dim + 2;
}

int TorqueBasedArmGenerator::get_output_dim(){
    assert(dim == 2);
    return 4*dim;
}
