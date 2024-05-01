/**
 * RNN.hpp | bmi_model
 *
 * Description:
 * -----------
 * This file contains:
 * 1. A simple struct for dealing with training parameters.
 * 2. The RNN class.
 */
#ifndef RNN_hpp
#define RNN_hpp

#include "Objectives.hpp"
#include "Monitor.hpp"
#include <fstream>
#include "Input.hpp"
#include "EligibilityTrace.hpp"
#include "Gradient.hpp"
#include "Target.hpp"
#include "globals.hpp"
#include "activations.hpp"
#include "VelocityKalmanFilter.hpp"

enum class OptimizerType {
    SGD,
    Adam,
};


/**
 * Simple struct for handling the training parameters of the RNN.
 */
struct TrainParamsRNN {
    
    int nb_epochs;                      // number of epochs to train on
    double learning_rate[4];            // learning rates of weight matrices U, W, V and bias b, in that order
    bool verbose;                       // whether to output training info to terminal
    int interval_verbose;               // number of epochs after which the training information is displayed
    bool learning_after_each_example;   // whether to learn after each example (true => SGD, false => batch)
    double scale_factor_hold;
    bool lr_adapt;
    double lr_adaptation_factor;
    
    TrainParamsRNN(int n, double lr[4], bool verbose, int interval_verbose,
                   bool learning_after_each_example, double scale_factor_hold=1.,
                   bool lr_adapt=true, double lr_adaptation_factor=0.99997) :
    nb_epochs(n), learning_rate{lr[0], lr[1], lr[2], lr[3]},
    verbose(verbose), interval_verbose(interval_verbose),
    learning_after_each_example(learning_after_each_example), scale_factor_hold(scale_factor_hold),
    lr_adapt(lr_adapt), lr_adaptation_factor(lr_adaptation_factor) {}
};


/**
 * Class for recurrent neural networks.
 *
 * At the moment, the only supported training method is REINFORCE.
 * Each unit performs a leaky integration of its total input, including noise.
 * Eligibility traces compute the correlation between the postsynaptic noise and the presynaptic activity.
 * Weight update is proportional to the reward-prediction error times the eligibility trace.
 */
class RNN {
    int _Nin, _Nrec, _Nout; // network topology
    
    double _noise;          // noise intensity
    double _alpha_reward;   // "forgetting" factor in computing the expected reward
    double _tau;            // membrane time constant
    
    double _a;              // dt / tau  (see globals.cpp for definition of integration time step `dt`)
    double _scale_initial_potential;
    
    ActivationType phi;
    
    std::vector<double> _long_reward_traces;  // keep reward traces in memory for a longer time (used under CLDA)
    std::vector<double> _long_reward_traces_preparatory;

    Matrix U, W, V; // input, recurrent and output weight matrices
    Vector b;       // bias vector for recurrent units
    
    Matrix m_U, m_W, v_U, v_W; // moments for Adam optimizer
    Vector m_b, v_b;
    Vector hold_signal;
    
    Vector v_init;
    std::mt19937 rng_for_dropout;

    
    /// Compute the gradient
    std::pair<GradientRecurrent, double> _compute_gradient(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int target_id, double *reward_trace, double *reward_traces_prep);
    
    /// Update gradient moments (for Adam optimizer)
    void update_moments(const GradientRecurrent& g, double beta1=0.9, double beta2=0.999);
    void reset_moments();
    
    /// Train all targets for a single epoch
    double _train_epoch(Loss * loss, Effector * imp,  Input & inputs, Target & targets, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[], int epoch, OptimizerType optimizer);
    double _train_epoch_with_dropout(Loss * loss, VelocityKalmanFilter &kf,  Input & inputs, Target & targets, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[], int epoch, OptimizerType optimizer, const double dropout_prob=0.1);
    
    /// Compute only the moving average of the reward, without computing gradients
    double _compute_reward_only(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[]);
    /// Whether the expected reward was memorized between calls to the `train` function (used with CLDA)
    bool _reward_was_memorized();
    
    /// Update weights using the gradients
    void _weight_update(double lr[4], const GradientRecurrent& g, int epoch, OptimizerType optimizer);
    
    /// Define initial membrane potential
    Vector find_initial_membrane_potential();
    double loss_for_v_init(const Vector &v);
    Vector set_initial_potential(const double scale=1.);
    
    
public:
    /// Constructor
    RNN(int Nin=8, int Nrec=64, int Nout=2, double noise=5e-3, double tau=5.e-2, double alpha_reward=0.3, ActivationType phi=ActivationType::tanh, double output_decimation=0.);
    
    /// Compute RNN's response to input `x`.
    Matrix forward(Input& x, int target_id); // w/o recordings
    Matrix forward(Input& x, int target_id, Monitor & m); // while recording RNN activity
    Matrix forward(Input& x, int target_id, Monitor & m_rec, Monitor & m_input); // while recording RNN activity and input
    Matrix forward(Input& x, int target_id, Monitor & m_rec, Monitor & m_input, Monitor & m_state); // while recording RNN activity, input and state

    /// Train RNN on `loss` using effector `imp` with `inputs` and `targets`.
    void train(Loss * loss, Effector * imp, Input & in, Target & t, TrainParamsRNN & p,
               std::string file_prefix, std::ofstream::openmode mode, OptimizerType optimizer=OptimizerType::Adam);
    void train_with_dropout(Loss * loss, VelocityKalmanFilter &kf, Input & in, Target & t, TrainParamsRNN & p,
                                 std::string file_prefix, std::ofstream::openmode mode, OptimizerType optimizer);
    
    /// Compute average reward
    double compute_mean_reward(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int nb_reals=20);
    double compute_median_reward(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int nb_reals=20);
    
    /// Defining reward traces that are kept in memory for a long time (typically used with CLDA)
    void init_long_reward_traces(int nb_targets);
    void reset_long_reward_traces();
    
    /// Transform parameters from tanh to sigmoid
    void transform_to_sigmoid();
    void transform_to_tanh();
    
    /// Getters
    Matrix getU() {return U;}
    Matrix getW() {return W;}
    Matrix getV() {return V;}
    Vector getb() {return b;}
    
    /// Setters
    void setU(const Matrix& new_U) {U = new_U;}
    void setW(const Matrix& new_W) {W = new_W;}
    void setV(const Matrix& new_V) {V = new_V;}
    void setb(const Matrix& new_b) {b = new_b;}
    
    /// Saving to file and loading from file
    void save(std::string file_prefix);
    void load(std::string file_prefix);
};

#endif /* RNN_hpp */
