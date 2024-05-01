/**
 * TorqueBasedArm.cpp | bmi_model
 */
#include "TorqueBasedArm.hpp"
#include "utilities.hpp"
#include <iostream>
#include "globals.hpp"

TorqueBasedArm::TorqueBasedArm(int dim, double tau_f,  double radius_for_reset): Effector(dim, radius_for_reset), _tau_f(tau_f) {
    effector_type = ARM;
    arm_type = TorqueBased;
    
    _l1 = 0.12;
    _l2 = 0.16;
    _m1 = 0.25;
    _m2 = 0.20;
    _I1 = 0.0018;
    _I2 = 0.0012;
    _d1 = 0.075;
    _d2 = 0.075;
    
    /*
     Parameters used typically:
     _l1 = 0.145;
     _l2 = 0.175;  //Lillicrap had 0.284
     _m1 = 0.211;
     _m2 = 0.194;
     _I1 = 0.025;  //=0.025 from Lillicrap; =0.00163 from Cheng and Scott
     _I2 = 0.045;  //=0.045 from Lillicrap; =0.00107 from Cheng and Scott
     _d1 = 0.075;
     _d2 = 0.075;
     */
    
    
    friction_matrix <<  0.05, 0.025,
                        0.025, 0.05;
    
    cartesian_center_position << 0, 0.2; //DEBUG !!! default 0, 0.15
    
    // Transform initial end_effector position to joint angles (angular vels and torques default to 0)
    Eigen::Vector2d init_joint_angles = transform_end_effector_position_to_joint_angles(cartesian_center_position, _l1, _l2);
    initial_state(0) = init_joint_angles(0);
    initial_state(1) = init_joint_angles(1);
    
    // Set initial end_effector
    end_effector_init_cond(0) = cartesian_center_position(0);
    end_effector_init_cond(1) = cartesian_center_position(1);
}


Eigen::Matrix2d TorqueBasedArm::inertia_matrix(const Eigen::Vector2d & angles) {
    double a1 = _I1 + _I2 + _m2 * _l1 * _l1;
    double a2 = _m2 * _l1 * _d2;
    double a3 = _I2;
    
    Eigen::Matrix2d M;
    M << a1 + 2*a2*cos(angles(1)), a3 + a2*cos(angles(1)),
    a3 + a2*cos(angles(1)), a3;
    
    return M;
}

Eigen::Vector2d TorqueBasedArm::get_end_effector_acceleration(const Eigen::VectorXd & s) {
    Eigen::Vector2d q;
    q << s(0), s(1);
    Eigen::Vector2d qdot;
    qdot << s(2), s(3);
    Eigen::Vector2d torques;
    torques << s(4), s(5);
    
    Eigen::Matrix2d dJdt = djacobiandt(q, qdot, _l1, _l2);
    Eigen::Matrix2d J = jacobian(q, _l1, _l2);
    Eigen::Matrix2d M = inertia_matrix(q);
    Eigen::Vector2d c = centripetal_and_Coriolis_force(q, qdot);
    
    return dJdt * qdot + J * inverse2d(M) * (torques - c - friction_matrix * qdot);
}


Eigen::Vector2d TorqueBasedArm::centripetal_and_Coriolis_force(const Eigen::Vector2d& angles, const Eigen::Vector2d& angular_velocities){
    double a2 = _m2 * _l1 * _d2;
    Eigen::Vector2d f;
    f(0) = -a2 * sin(angles(1)) * angular_velocities(1) * (2 * angular_velocities(0) + angular_velocities(1));
    f(1) = a2 * sin(angles(1)) * angular_velocities(0) * angular_velocities(0);
    return f;
}

Eigen::MatrixXd TorqueBasedArm::get_end_effector_trajectory(const Eigen::MatrixXd & ctrls){
    reset();
    long total_time = ctrls.cols();
    Eigen::MatrixXd s = Eigen::MatrixXd::Zero(3*_dim, total_time);
    
    Eigen::Vector2d angles;
    angles << state(0), state(1);
    Eigen::Vector2d ang_vels;
    ang_vels << state(2), state(3);
    Eigen::Vector2d torques;
    torques << state(4), state(5);
    
    
    for (int t(0);t<total_time;t++){
        Eigen::Matrix2d M = inertia_matrix(angles);
        Eigen::Vector2d c = centripetal_and_Coriolis_force(angles, ang_vels);
        
        angles = angles + dt * ang_vels;
        ang_vels = ang_vels + dt * inverse2d(M) * (torques - c - friction_matrix * ang_vels);
        torques = torques + dt * (ctrls.col(t) - torques)/_tau_f;
        Eigen::VectorXd angular_state(3*_dim);
        angular_state << angles(0), angles(1), ang_vels(0), ang_vels(1), torques(0), torques(1);
        
        Eigen::Vector2d pos = transform_joint_angles_to_end_effector_position(angles, _l1, _l2);
        Eigen::Vector2d vel = transform_joint_velocity_to_end_effector_velocity(angles, ang_vels, _l1, _l2);
        //Eigen::Vector2d f = transform_joint_torque_to_end_effector_force(angles, torques, _l1, _l2);
        Eigen::Vector2d acc = get_end_effector_acceleration(angular_state);
        
        s(0, t) = pos(0) - cartesian_center_position(0);
        s(1, t) = pos(1) - cartesian_center_position(1);
        s(2, t) = vel(0);
        s(3, t) = vel(1);
        s(4, t) = acc(0);
        s(5, t) = acc(1);
    }
    return s;
}

void TorqueBasedArm::next_state(const Eigen::VectorXd& control){
    Eigen::Vector2d angles;
    angles << state(0), state(1);
    Eigen::Vector2d ang_vels;
    ang_vels << state(2), state(3);
    Eigen::Vector2d torques;
    torques << state(4), state(5);
    
    Eigen::Matrix2d M = inertia_matrix(angles);
    Eigen::Vector2d c = centripetal_and_Coriolis_force(angles, ang_vels);
    
    angles = angles + dt * ang_vels;
    ang_vels = ang_vels + dt * inverse2d(M) * (torques - c - friction_matrix * ang_vels);
    torques = torques + dt * (control - torques)/_tau_f;
    
    state(0) = angles(0); state(1) = angles(1);
    state(2) = ang_vels(0); state(3) = ang_vels(1);
    state(4) = torques(0); state(5) = torques(1);
    
    end_effector = transform_state_to_end_effector(state);
}

void TorqueBasedArm::reset(){
    state = initial_state;
    end_effector = end_effector_init_cond;
    
    if (radius_for_reset > 1e-6) {
        std::uniform_real_distribution<double> unif_radius(0., radius_for_reset);
        std::uniform_real_distribution<double> unif_angle(0., 2*M_PI);
        
        double r{ unif_radius(rng_for_reset) };
        double angle{ unif_angle(rng_for_reset) };
        
        end_effector(0) = cartesian_center_position(0) + r * cos(angle);
        end_effector(1) = cartesian_center_position(1) + r * sin(angle);
        
        Eigen::Vector2d init_end_effector_pos;
        init_end_effector_pos << end_effector(0), end_effector(1);
        
        Eigen::Vector2d init_joint_angles = transform_end_effector_position_to_joint_angles(init_end_effector_pos, _l1, _l2);
        state(0) = init_joint_angles(0);
        state(1) = init_joint_angles(1);
    }
}

Eigen::VectorXd TorqueBasedArm::transform_state_to_end_effector(const Eigen::VectorXd& state){
    Eigen::VectorXd end_effector = state;
    
    Eigen::Vector2d angles;
    angles << state(0), state(1);
    Eigen::Vector2d x = transform_joint_angles_to_end_effector_position(angles, _l1, _l2);
    end_effector(0) = x(0) - cartesian_center_position(0);
    end_effector(1) = x(1) - cartesian_center_position(1);

    Eigen::Vector2d angular_velocities;
    angular_velocities << state(2), state(3);
    x = transform_joint_velocity_to_end_effector_velocity(angles, angular_velocities, _l1, _l2);
    end_effector(2) = x(0);
    end_effector(3) = x(1);

    
    //Eigen::Vector2d torques;
    //torques << state(4), state(5);
    x = get_end_effector_acceleration(state);
    //x = transform_joint_torque_to_end_effector_force(angles, torques, _l1, _l2);
    end_effector(4) = x(0);
    end_effector(5) = x(1);
    
    return end_effector;
}
