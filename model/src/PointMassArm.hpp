/**
 * PointMassArm.hpp | bmi_model
 *
 * Description:
 * -----------
 * A simple point mass is displaced by a linear force applied to it.
 * The force itself is being controlled by the network.
 */

#ifndef PointMassArm_hpp
#define PointMassArm_hpp
#include <Eigen/Dense>
#include "Effector.hpp"

class PointMassArm : public Effector {
    double _tau_f;    // time constant of force
    double _mass;     // mass of the "arm"
   
public:
    Eigen::MatrixXd A, B;
    
    PointMassArm(int dim=2, double tau_f=0.04, double mass=1., double radius_for_reset=0.);
    
    Eigen::MatrixXd get_end_effector_trajectory(const Eigen::MatrixXd & controls);
    void next_state(const Eigen::VectorXd& control);
    void reset();
    int get_nb_channels(){ return 0; }
    Effector* get_copy(){return new PointMassArm(*this);};

};


#endif /* PointMassArm_hpp */
