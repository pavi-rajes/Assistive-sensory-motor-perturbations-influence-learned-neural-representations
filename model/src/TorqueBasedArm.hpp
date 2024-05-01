/**
 * TorqueBasedArm | bmi_model
 *
 * Description:
 * -----------
 * 2D planar torque-based arm.
 */
#ifndef TorqueBasedArm_hpp
#define TorqueBasedArm_hpp

#include "Effector.hpp"
#include <Eigen/Dense>

class TorqueBasedArm : public Effector{

    double _l1, _l2;  // length for first and second link (m)
    double _m1, _m2;  // mass of each link (kg)
    double _I1, _I2;  // moment of inertia (kg.m**2)
    double _d1, _d2;  // distance from joint center to center of mass (m)
    double _tau_f;    // time constant of torque
    
    Eigen::Matrix2d friction_matrix;
    Eigen::Vector2d cartesian_center_position;
    
    Eigen::Matrix2d inertia_matrix(const Eigen::Vector2d & angles);
    Eigen::Vector2d centripetal_and_Coriolis_force(const Eigen::Vector2d & angles, const Eigen::Vector2d & angular_velocities);
    
public:
    TorqueBasedArm(int dim=2, double tau_f=0.04, double radius_for_reset=0.);
    
    /**
     Compute the (Cartesian) end_effector position.
     @param ctrls activating the arm
     */
    Eigen::MatrixXd get_end_effector_trajectory(const Eigen::MatrixXd & ctrls);
    
    Eigen::VectorXd transform_state_to_end_effector(const Eigen::VectorXd& state);
    
    Eigen::Vector2d get_end_effector_acceleration(const Eigen::VectorXd & state);
    
    void next_state(const Eigen::VectorXd& control);
    void reset();
    
    Effector* get_copy(){
        return new TorqueBasedArm(*this);
    };
    
    /// Getters
    double get_l1(){ return _l1; }
    double get_l2(){ return _l2; }
    Eigen::Vector2d get_center_position(){ return cartesian_center_position; }
    int get_nb_channels(){ return 0; }
};


#endif /* TorqueBasedArm_hpp */
