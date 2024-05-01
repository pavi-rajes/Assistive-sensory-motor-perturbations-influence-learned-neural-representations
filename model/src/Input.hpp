/**
 * Input.hpp | bmi_model
 *
 * Contains classes pertaining to the input applied to the network.
 *
 * Brief overview of the classes:
 *  -        Input : Abstract base class
 *  -  StaticInput : Input whose value does not depend on the effector's state (no feedback).
 *  - DynamicInput : Input whose value does depend on sensor feedback of the effector.
 *  -     FFNInput : Input that uses a feedforward network (FFN) to encode visual information about the target, context and (optionally) the effector's state.
 *  -    DualInput : (Rarely used) Input that may contain both proprioceptive and visual information (hence Dual).
 */
#ifndef Input_hpp
#define Input_hpp

#include <Eigen/Dense>
#include <vector>
#include "Effector.hpp"
#include "FFN.hpp"
#include <stdexcept>
#include <deque>

enum InputType {NOINPUT, STATIC, DYNAMIC, FFNINPUT, DUALINPUT};  //TODO: should be scoped enum


class Input {
    
protected:
    bool include_context_bit;   // whether a "context bit" is included in the input
    int nb_targets;             // number of targets
    int duration;               // motion duration, from go cue to end-of-trial
    int hold_duration;          // motion duration, from go cue to end-of-trial
    InputType input_type;       // type of input

public:
    Input(int nb_targets=8, bool include_context_bit=false, int duration=50, int hold_duration=0);
    
    Effector *imp;
    virtual Eigen::VectorXd operator()(int i) =0;
    
    /**
     * Compute the next input.
     *
     * Parameters:
     *  - controls = output of the RNN
     *  - activities = RNN activities
     *  - target_dir = unit-norm target direction (used in conjunction with PassiveCursor)
     */
    virtual void next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir) =0;
    virtual void set_context_bit(int&& ) =0;
    virtual void reset() =0;
    virtual ~Input() {}
    
    void set_context(Effector *imp) {this->imp = imp;}
    EffectorType get_context() {return imp->effector_type;};
    int get_nb_targets() {return nb_targets;}
    int get_duration() {return duration;}
    int get_hold_duration(){return hold_duration;}
    InputType get_input_type() {return input_type;}
};


// STATIC INPUT
class StaticInput: public Input {

public:
    std::vector<Eigen::VectorXd> inputs;

    StaticInput(Effector *imp, int nb_targets=8, bool include_context_bit=false, int duration=50, int hold_duration=0);
    void next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir);
    Eigen::VectorXd operator()(int i);
    void set_context(Effector *imp);
    void reset();
    ~StaticInput() {}
};


// DYNAMIC INPUT
class DynamicInput: public Input {
    double distance_to_target;
    
public:
    DynamicInput(Effector *imp, int nb_targets=8, bool include_context_bit=false,
                 int duration=50, double distance_to_target=0.07, int hold_duration=0);
    void next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir);
    Eigen::VectorXd operator()(int i);
    void set_context_bit(int&& b) {std::cout << "Cannot set context bit manually for DynamicInput. Ignored.\n";};
    void reset();
    EffectorType get_context() { return imp->effector_type; }
    double get_distance_to_target() {return distance_to_target;}
    int input_size();
};


// FFN INPUT
class FFNInput: public Input {
    double distance_to_target;
    FFN *ffn;
    std::deque<Eigen::VectorXd> buffer;
    int feedback_delay;
public:
    FFNInput(Effector *imp, FFN *ffn, int nb_targets=8, bool include_context_bit=false,
             int duration=50, double distance_to_target=0.07, int hold_duration=10, int feedback_delay=0);
    void next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir);
    Eigen::VectorXd operator()(int i);
    void set_context_bit(int&& b) {std::cout << "Cannot set context bit manually for FFNInput. Ignored.\n";};
    void reset();
    int input_size();
    void initialize_buffer(int i);
    Eigen::MatrixXd get_U();
};


// DUAL INPUT (PROPRIOCEPTIVE AND VISUAL, used for instance with TorqueBasedArm)
class DualInput: public Input {
    double distance_to_target;
    FFN *ffn;
    Effector *arm;
    
public:
    DualInput(Effector *imp, Effector *arm, FFN *ffn, int nb_targets=8, bool include_context_bit=false,
             int duration=50, double distance_to_target=0.07, int hold_duration=10);
    void next(const Eigen::VectorXd& controls, const Eigen::VectorXd& activities, const Eigen::VectorXd& target_dir);
    Eigen::VectorXd operator()(int i);
    void set_context_bit(int&& b) {std::cout << "Cannot set context bit manually for DualInput. Ignored.\n";};
    void reset();
    int input_size();
    Eigen::MatrixXd get_U();
};


#endif /* Input_hpp */
