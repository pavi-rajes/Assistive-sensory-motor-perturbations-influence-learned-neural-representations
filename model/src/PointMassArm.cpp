/**
 * PointMassArm.cpp | bmi_model
 */
#include "PointMassArm.hpp"
#include "globals.hpp"
#include <iostream>
using namespace Eigen;

PointMassArm::PointMassArm(int dim, double tau_f, double mass, double radius_for_reset): Effector(dim, radius_for_reset), _tau_f(tau_f), _mass(mass) {
    effector_type = ARM;
    arm_type = PointMass;
    
    A = MatrixXd::Identity(3*dim, 3*dim);
    for(int i(0); i<_dim; i++) A(i, i + dim) = dt;
    for(int i(_dim); i<2*_dim;i++) A(i, i + dim) = dt/_mass;
    for(int i(2*_dim); i<3*_dim;i++) A(i, i) = exp(-dt/_tau_f);
    
    
    B = MatrixXd::Zero(3*dim, 2);
    B(2*dim, 0) = 1.;
    B(2*dim+1, 1) = 1.;
}

MatrixXd PointMassArm::get_end_effector_trajectory(const MatrixXd & ctrls){
    reset();
    int total_time = int(ctrls.cols());
    MatrixXd s(3*_dim, total_time);
    
    s.col(0) = A*state + B*ctrls.col(0);
    for (int t(1);t<total_time;t++){
        s.col(t) = A*s.col(t-1) + B*ctrls.col(t-1);
    }
    return s;
}

void PointMassArm::next_state(const Eigen::VectorXd& control) {
    state = A*state + B*control;
    end_effector = state;
}

void PointMassArm::reset() {
    state = initial_state;
    end_effector = end_effector_init_cond;
    if (radius_for_reset > 1e-6) {
        std::uniform_real_distribution<double> unif_radius(0., radius_for_reset);
        std::uniform_real_distribution<double> unif_angle(0., 2*M_PI);
        
        double r{ unif_radius(rng_for_reset) };
        double angle{ unif_angle(rng_for_reset) };
        
        end_effector(0) = r * cos(angle);
        end_effector(1) = r * sin(angle);
        state = end_effector;
    }
}


