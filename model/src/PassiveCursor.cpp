/**
 * PassiveCursor.cpp | bmi_model
 */
#include "PassiveCursor.hpp"
#include <iostream>

PassiveCursor::PassiveCursor(int duration, double distance_to_target, int dim): Effector(dim), duration(duration), distance_to_target(distance_to_target) {
    effector_type = PASSIVECURSOR;
    arm_type = None;
}

Matrix PassiveCursor::get_end_effector_trajectory(const Matrix & unit_target_directions){
    reset();
    int total_time = int(unit_target_directions.cols());
    Matrix s = Matrix::Zero(3*_dim, total_time);
    Vector target_dir(_dim);
    for (int i=0;i<_dim;i++) target_dir(i) = unit_target_directions(i,0);
        
    for (int i=0;i<_dim;i++) s(i,0) = state(i) + distance_to_target*target_dir(i)/duration;
    for (int i=0;i<_dim;i++) s(i+_dim,0) = distance_to_target*target_dir(i)/(duration*dt);
    
    for (int t(1);t<total_time;t++){
        for (int i=0;i<_dim;i++) s(i,t) = s(i,t-1) + distance_to_target*target_dir(i)/duration;
        for (int i=0;i<_dim;i++) s(i+_dim,t) = distance_to_target*target_dir(i)/(duration*dt);
    }
    return s;
}

void PassiveCursor::next_state(const Vector& unit_target_direction) {
    for (int i=0;i<_dim;i++) state(i) = state(i) + distance_to_target*unit_target_direction(i) / duration;
    for (int i=0;i<_dim;i++) state(i+_dim) = distance_to_target*unit_target_direction(i) /(duration*dt);
    end_effector = state;
}

void PassiveCursor::reset() {
    state = initial_state;
    end_effector = end_effector_init_cond;
}
