/**
 * File SIM_ARM_MODEL.cpp
 *
 * Description:
 * -----------
 * Test: Control the torque-based arm under closed-loop feedback control.
 */
#include "rnn4bci.hpp"
using namespace std;


int main(int argc, char *argv[]) {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                            Command line options                               //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    cout << "Simulation name : " + string(argv[0]) << endl;
    int narg = 1;
    
    string dir;
    int n_targets{ 8 };         // number of targets to reach for
    int n_reals{ 10 };          // number of samples to generate when outputting data to file and fitting the BMI decoder
    int hold_duration{ 25 };    // duration of hold/preparatory period (before go cue)
    
    while (narg < argc) {
        string option = string(argv[narg]);
        stringstream ss{argv[narg+1]};
        if (option == "--seed" || option == "-s") {
            ss >> seed_matrix;
            myrng.seed(seed_matrix);
            cout << "Seed for weight matrix : " << seed_matrix << '\n';
        } else if (option == "--dir") {
            dir = string(argv[narg+1]);
            cout << "Output directory : " << dir << '\n';
        } else if (option == "--n_targets") {
            ss >> n_targets;
            cout << "Number of targets : " << n_targets << '\n';
        } else if (option == "--n_reals") {
            ss >> n_reals;
            cout << "Number of realizations to output for kinematics : " << n_reals << '\n';
        } else if (option == "--hold_duration") {
            ss >> hold_duration;
            cout << "Hold duration during preparation : " << hold_duration << '\n';
        }
        else {
            cerr << "Option " << argv[narg] << " is unknown and will be ignored." << '\n';
        }
        narg += 2;
    }
    cout << "Integration time step : " << dt << endl << endl;

    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                                  Parameters                                   //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    constexpr int n_rec{ 100 };                     // number of recurrent units
    constexpr int n_out{ 2 };                       // number of output units
    constexpr int dim{ 2 };                         // workspace dimension
    constexpr double distance_to_target{ 0.07 };    // distance of target from center of workspace
    constexpr int duration{ 100 };                  // duration of motion from go cue (in units of integration time step)
    constexpr bool use_context_bit{ true };         // deprecated (kept for backward compatibility)
    constexpr double gamma_f{ 0.05 };               // hyperparameter for regularizer on force             //DEBUG!!!!!!!!!!!!!!!!!!
    constexpr double noise_amplitude{ 2e-2 };       // amplitude of private noise applied to neurons     (default: 2e-2)  //DEBUG!!!!!!!!!!!!!!!!!!
    constexpr double tau_m{ 5.e-2 };                // membrane time constant (default: 5e-2) //DEBUG!!!!!!
    constexpr double alpha_reward{ 0.3 };           // filtering factor for reward traces
    constexpr double activity_regularizer{ 0. };   // 0.25
    ActivationType phi = ActivationType::relu;      // chosen activation function
    const int feedback_delay{ int(0.1/dt) };
    constexpr double output_decimation{ 0. };
    
    constexpr double effort_penalty_manual{ 5.e2 };    // hyperparameter for effort penalty (sq norm of manual controls)
    constexpr double gamma_v_manual{ 0.25 };            // hyperparameter for regularizer on velocity (typical: 0.1)
    constexpr double scale_factor_hold_manual{ 1. };    // scaling factor for the "hold-at-center" part of the reach
    constexpr bool rnn_is_pretrained{ false };          // whether the pre-BMI learning network has been already pretrained
    constexpr bool ffn_is_pretrained{ false };           // whether the FFN-based input network has been already trained
    constexpr int nb_epochs{ int(3e5) };                // Typical: int(2e5)
    constexpr double lr{ 2e-5 };                       // Typical: [1e-5, 5e-5]; default 2e-5
    double learning_rates[4] = {lr, lr, 0, lr};
    constexpr bool learn_after_each_ex{ false };
    constexpr bool adapt_learning_rate{ true };
    constexpr bool verbose{ true };
    constexpr int interval_verbose{ 5000 };
    constexpr double learning_rate_adaptation{ 0.999999 };
    TrainParamsRNN p_manual_before(nb_epochs, learning_rates, verbose, interval_verbose,
                                   learn_after_each_ex, scale_factor_hold_manual, adapt_learning_rate, learning_rate_adaptation);
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                                  Simulation                                   //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Pre-train input FFN
    TorqueBasedArm arm(2, 0.04, 0.005);
    constexpr double max_normalized_distance{ 5. };
    MixedGenerator tg(dim, max_normalized_distance);

    const unsigned int input_dim = tg.get_input_dim();
    const unsigned int output_dim = tg.get_output_dim();
    const int rnn_input_layer_size{ n_rec };
    FFN ffn({input_dim, rnn_input_layer_size, output_dim}, phi);
    if (ffn_is_pretrained) {
        ffn.load("../data/torque-closedloop/random-ics-arm-no-shallow-ffn-train-no-decimation-N200/seed"+to_string(seed_matrix)+"/ffn");
        ffn.set_input_type(tg.get_datatype());
    }
    else {
        ffn.set_input_type(tg.get_datatype());
        //ffn.train(tg, 500, 1e-4, 1e-4, 0.999, 100); // train for 100 epochs, lr_hidden = 2e-4, lr_output = 2e-4, Adam beta2 = 0.999, batch_size = 100
    }
    ffn.save(dir+"ffn");
    Eigen::MatrixXd U = ffn.get_U();
    
    // Effector, inputs and targets
    FFNInput inputs(&arm, &ffn, n_targets, use_context_bit, duration, distance_to_target, hold_duration, feedback_delay);
    Target targets(dim, n_targets, distance_to_target);
    
    // Network
    const int n_in{ inputs.input_size() };
    RNN net(n_in, n_rec, n_out, noise_amplitude, tau_m, alpha_reward, phi, output_decimation);
    if (rnn_is_pretrained) net.load("../data/torque-closedloop/random-ics-arm-no-shallow-ffn-train-no-decimation-N200/seed"+to_string(seed_matrix)+"/manual_trained_network");
    
    
    // Training objectives
    Loss * loss_manual = new EndLoss(gamma_v_manual, gamma_f, effort_penalty_manual, activity_regularizer);
    
    
    if (not rnn_is_pretrained) {
        net.save(dir+"naive_network");
        save_to_file(dir, "naive", net, &arm, inputs, n_reals);
    }
    
    //=================================================================================//
    //                  PART 1: Pretraining under manual control                       //
    //=================================================================================//
    //if (not rnn_is_pretrained){
        // Training under manual control
        cout << "Training under manual control..." << endl;
        
        net.train(loss_manual, &arm, inputs, targets, p_manual_before, dir+"manual_trained_network_", ofstream::out, OptimizerType::Adam);
        
        cout << "End manual training" << endl << endl;
    //}
    // Save data for initial manual control
    net.save(dir+"manual_trained_network");
    save_to_file(dir, "manual_before_learning", net, &arm, inputs, n_reals);
    
    // Delete
    delete loss_manual;
    
    return 0;
}

