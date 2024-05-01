/**
 * Input.cpp | bmi_model
 */
#include "Input.hpp"
#include <assert.h>
#include <iostream>
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                  Virtual Input class                  //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
Input::Input(int nb_targets, bool include_context_bit, int duration, int hold_duration): nb_targets(nb_targets), include_context_bit(include_context_bit), duration(duration), input_type(NOINPUT), imp(nullptr), hold_duration(hold_duration) {}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                      StaticInput                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
StaticInput::StaticInput(Effector *_imp, int nb_targets, bool include_context_bit, int duration, int hold_duration): Input(nb_targets, include_context_bit, duration, hold_duration) {
    input_type=STATIC;
    imp = _imp;
    inputs.reserve(nb_targets);

    for (int i(0);i<nb_targets;++i){
        Eigen::VectorXd temp;
        if (include_context_bit) temp = Eigen::VectorXd::Zero(nb_targets+1);
        else temp = Eigen::VectorXd::Zero(nb_targets);
        temp(i) = 1.; //context bit set to zero by default
        inputs.push_back(temp);
    }
}
void StaticInput::reset() {
    imp->reset();
}

void StaticInput::next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir){
    // Even though the input is static, still need to evolve the effector
    if (imp->effector_type==DECODER) imp->next_state(activities);
    else if (imp->effector_type==ARM) imp->next_state(controls);
    else if (imp->effector_type==PASSIVECURSOR) imp->next_state(target_dir);
}

Eigen::VectorXd StaticInput::operator()(int i){
    return inputs[i];
}

void StaticInput::set_context(Effector *imp){
    if (include_context_bit) {
        if (imp->effector_type == ARM){
            for (int i(0);i<nb_targets;++i) inputs[i](nb_targets) = 1.;
        }
        else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
            for (int i(0);i<nb_targets;++i) inputs[i](nb_targets) = 0.;
        }
    }
    else std::cerr << "Can't change context when there is not context bit.\n";
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                      DynamicInput                     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
DynamicInput::DynamicInput(Effector *_imp, int nb_targets, bool include_context_bit, int duration, double distance_to_target, int hold_duration) : Input(nb_targets, include_context_bit, duration, hold_duration), distance_to_target(distance_to_target) {
    input_type=DYNAMIC;
    imp = _imp;
}

void DynamicInput::next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir){
    if (imp->effector_type==DECODER) imp->next_state(activities);
    else if (imp->effector_type==ARM) imp->next_state(controls);
    else if (imp->effector_type==PASSIVECURSOR) imp->next_state(target_dir);
}

void DynamicInput::reset() {
    imp->reset();
}

Eigen::VectorXd DynamicInput::operator()(int i){
    Eigen::VectorXd x(input_size());
    
    if (imp->effector_type == ARM){
        if (imp->arm_type == PointMass) {
            if (imp->get_dim() == 2) {
                x << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1.;
            } else {
                throw std::out_of_range("Dimension other than 2 not supported. Exiting.");
            }
            
        } else if (imp->arm_type == TorqueBased) {
            if (imp->get_dim() == 2) {
                x << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, imp->state(0), imp->state(1), imp->state(2), imp->state(3), imp->state(4), imp->state(5), cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1;
            } else {
                throw std::out_of_range("Dimension other than 2 not supported. Exiting.");
            }
        }
    } else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
        if (imp->get_dim() == 2) {
            x << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0.;
        } else {
            throw std::out_of_range("Dimension other than 2 not supported. Exiting.");
        }
    }
    return x;
}

int DynamicInput::input_size(){
    int s(0);
    
    if (imp->effector_type == ARM){
        if (imp->arm_type == PointMass) {
            s = 2*imp->get_dim() + 1;
        } else if (imp->arm_type == TorqueBased) {
            if (imp->get_dim() == 2) {
                s = (3+1+1)*imp->get_dim();
            }
            else {
                throw std::runtime_error("Torque-based arm should have dim = 2.");
            }
        }
    } else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
        s = 2*imp->get_dim() + 1;
    }
    return s;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                      FFNInput                         //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
FFNInput::FFNInput(Effector *imp, FFN *ffn, int nb_targets, bool include_context_bit, int duration, double distance_to_target, int hold_duration, int feedback_delay) : Input(nb_targets, include_context_bit, duration, hold_duration), distance_to_target(distance_to_target), feedback_delay(feedback_delay) {
    input_type=FFNINPUT;
    this->imp = imp;
    this->ffn = ffn;
    
    for (int i=0; i<=feedback_delay; i++) buffer.push_back(Eigen::VectorXd::Zero(ffn->get_input_size()));
}

void FFNInput::next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir){
    if (imp->effector_type==DECODER) imp->next_state(activities);
    else if (imp->effector_type==ARM) imp->next_state(controls);
    else if (imp->effector_type==PASSIVECURSOR) {
        imp->next_state(target_dir);
    }
}

void FFNInput::reset() {
    imp->reset();
    buffer.clear();
}

int FFNInput::input_size(){
    return ffn->get_output_size_to_RNN();
}

void FFNInput::initialize_buffer(int i){
    Eigen::VectorXd input_to_FFN(ffn->get_input_size());
    
    if (imp->effector_type == ARM){
        if (imp->arm_type == PointMass or imp->arm_type == TorqueBased) {
            if (ffn->get_input_type() == DataGenerator::target_and_position){
                input_to_FFN << imp->end_effector_init_cond(0)/distance_to_target, imp->end_effector_init_cond(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1., 0.;
            } else if (ffn->get_input_type() == DataGenerator::target_only) {
                input_to_FFN << cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1., 0.;
            }
        } else {
            std::cerr << "Wrong arm selection for FFNInput." << std::endl;
            exit(EXIT_FAILURE);
        }
        
    } else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
        if (ffn->get_input_type() == DataGenerator::target_and_position){
            input_to_FFN << imp->end_effector_init_cond(0)/distance_to_target, imp->end_effector_init_cond(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0., 1.;
        } else if (ffn->get_input_type() == DataGenerator::target_only) {
            input_to_FFN << cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0., 1.;
        }
    }
    
    for (int t=0; t<=feedback_delay; t++) buffer.push_back(input_to_FFN);
}


Eigen::VectorXd FFNInput::operator()(int i){
    assert(imp->get_dim() == 2);
    assert(ffn->get_input_type() != DataGenerator::unspecified);
    assert(ffn->get_input_type() != DataGenerator::torque_based_arm);
    
    Eigen::VectorXd input_to_FFN(ffn->get_input_size());
    
    // Construct initial buffer state, if empty (i.e., at the start of a trial)
    if (buffer.empty()) initialize_buffer(i);
    
    // Compute
    if (imp->effector_type == ARM){
        if (imp->arm_type == PointMass or imp->arm_type == TorqueBased) {
            if (ffn->get_input_type() == DataGenerator::target_and_position){
                input_to_FFN << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1., 0.;
            } else if (ffn->get_input_type() == DataGenerator::target_only) {
                input_to_FFN << cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1., 0.;
            }
        } else {
            std::cerr << "Wrong arm selection for FFNInput." << std::endl;
            exit(EXIT_FAILURE);
        }
        
    } else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
        if (ffn->get_input_type() == DataGenerator::target_and_position){
            input_to_FFN << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0., 1.;
        } else if (ffn->get_input_type() == DataGenerator::target_only) {
            input_to_FFN << cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0., 1.;
        }
    }
    
    buffer.push_back(input_to_FFN);
    input_to_FFN = buffer.front();
    buffer.pop_front();
    return ffn->penultimate_hidden_layer_activation(input_to_FFN);
}

Eigen::MatrixXd FFNInput::get_U(){
    return ffn->get_U();
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//                      DualInput                        //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
DualInput::DualInput(Effector *imp, Effector *arm, FFN *ffn, int nb_targets, bool include_context_bit, int duration, double distance_to_target, int hold_duration) : Input(nb_targets, include_context_bit, duration, hold_duration), distance_to_target(distance_to_target) {
    input_type=DUALINPUT;
    this->arm = arm;
    this->imp = imp;
    this->ffn = ffn;
}

void DualInput::next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir){
    if (imp->effector_type==DECODER) imp->next_state(activities);
    else if (imp->effector_type==ARM) imp->next_state(controls);
    else if (imp->effector_type==PASSIVECURSOR) imp->next_state(target_dir);
    arm->next_state(controls);
}

void DualInput::reset() {
    imp->reset();
    arm->reset();
}

int DualInput::input_size(){
    return ffn->get_output_size_to_RNN();
}

Eigen::VectorXd DualInput::operator()(int i){
    assert(imp->get_dim() == 2);
    assert(ffn->get_input_type() != DataGenerator::unspecified);
    
    Eigen::VectorXd x(input_size());
    Eigen::VectorXd input_to_FFN(ffn->get_input_size());
    
    if (imp->effector_type == ARM){
        if (imp->arm_type == TorqueBased) {
            if (ffn->get_input_type() == DataGenerator::torque_based_arm){
                input_to_FFN << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, arm->state(0)/M_PI, arm->state(1)/M_PI, arm->state(2)/M_PI, arm->state(3)/M_PI, arm->state(4), arm->state(5), cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 1., 0.;
            }
            else {
                std::cerr << "Wrong arm selection for DualInput." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        
    } else if (imp->effector_type == DECODER or imp->effector_type == PASSIVECURSOR) {
        if (ffn->get_input_type() == DataGenerator::torque_based_arm){
            input_to_FFN << imp->end_effector(0)/distance_to_target, imp->end_effector(1)/distance_to_target, arm->state(0)/M_PI, arm->state(1)/M_PI, arm->state(2)/M_PI, arm->state(3)/M_PI, arm->state(4), arm->state(5), cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets)), 0., 1.;
        }
    }
    x = ffn->penultimate_hidden_layer_activation(input_to_FFN);
    return x;
}

Eigen::MatrixXd DualInput::get_U(){
    return ffn->get_U();
}
