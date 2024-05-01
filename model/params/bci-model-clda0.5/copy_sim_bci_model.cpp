/**
 * File SIM_BCI_MODEL.cpp
 *
 * Description:
 * -----------
 * Simulation of manual and BMI control under a closedloop, feedforward-network-mediated input,
 * closed-loop decoder adaptation and unit swapping.
 * Technically, this file could be used to simulate both arm reaching and BCI (cf. parameters that refer to `manual` control).
 * We found it easier (for debugging and fine-tuning) to separate the two simulations. Hence, you can ignore all reference to manual parameters here.
 */
#include "rnn4bci.hpp"
using namespace std;

void evaluate_compactness(const string& dir, const string &file_suffix, int n_readouts, RNN &net, Loss * loss, const VelocityKalmanFilter &kf, Input & in, Target & target, TrainParamsRNN & p, int n_reals=10);

int main(int argc, char *argv[]) {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                            Command line options                               //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    cout << "Simulation name : " + string(argv[0]) << endl;
    int narg = 1;
    string dir;
    double alpha_CLDA{ 1. };    //(1 => no CLDA)
    int n_targets{ 8 };         // number of targets to reach for
    int n_reals{ 5 };           // number of samples to generate when outputting data to file and fitting the BMI decoder
    int hold_duration{ 25 };    // duration of hold/preparatory period (before go cue)
    cout << '\n';
    
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
        } else if (option == "--clda") {
            ss >> alpha_CLDA;
            cout << "CLDA : " << alpha_CLDA << '\n';
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
    cout << "Integration time step : " << dt << "\n\n";

    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                                  Parameters                                   //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Common task parameters
    constexpr int n_rec{ 100 };                     // number of recurrent units
    constexpr int n_out{ 2 };                       // number of output units
    constexpr int n_readouts{ 12 };                 // number of readout units for BMI control
    constexpr int dim{ 2 };                         // workspace dimension
    constexpr double distance_to_target{ 0.07 };    // distance of target from center of workspace
    constexpr int duration{ 100 };                  // duration of motion from go cue (in units of integration time step)
    constexpr bool use_context_bit{ true };         // deprecated (kept for backward compatibility)
    constexpr double gamma_f{ 0.05 };               // hyperparameter for regularizer on force
    constexpr double noise_amplitude{ 2e-2 };       // amplitude of private noise applied to neurons     (default: 2e-2)
    constexpr double tau_m{ 5.e-2 };                // membrane time constant (default: 5e-2)
    constexpr double alpha_reward{ 0.3 };           // filtering factor for reward traces
    constexpr double activity_regularizer{ 0.0 };   // 0.25
    ActivationType phi = ActivationType::relu;      // chosen activation function
    const int feedback_delay{ int(0.1/dt) };
    constexpr double radius_for_reset{ 0.005 };
    
    
    // Manual task parameters
    constexpr double effort_penalty_manual{ 0.e-3 };    // hyperparameter for effort penalty (sq norm of manual controls)
    constexpr double gamma_v_manual{ 0.25 };            // hyperparameter for regularizer on velocity (typical: 0.1)
    constexpr double scale_factor_hold_manual{ 1. };    // scaling factor for the "hold-at-center" part of the reach
    constexpr bool rnn_is_pretrained{ true };           // whether the pre-BMI learning network has been already pretrained
    constexpr bool ffn_is_pretrained{ true };           // whether the FFN-based input network has been already trained

    // Manual-pretraining-specific parameters
    constexpr int nb_epochs{ int(3e5) };                // Typical: int(2e5)
    constexpr double lr{ 2e-5 };                        // Typical: [1e-5, 5e-5]
    double learning_rates[4] = {lr, lr, 0, lr};
    constexpr bool learn_after_each_ex{ false };
    constexpr bool adapt_learning_rate{ true };
    constexpr bool verbose{ true };
    constexpr int interval_verbose{ 5000 };
    constexpr double learning_rate_adaptation{ 0.999999 };
    TrainParamsRNN p_manual_before(nb_epochs, learning_rates, verbose, interval_verbose,
                                   learn_after_each_ex, scale_factor_hold_manual, adapt_learning_rate, learning_rate_adaptation);
    
    // Continual BMI learning task parameters
    constexpr int nb_bmi_epochs{ 100 };
    constexpr int nb_manual_epochs{ 50 };
    constexpr int nb_days{ 25 };
    
    constexpr double effort_penalty_bmi{ 0e3 };
    constexpr double gamma_v_bmi{ 0.1 };
    constexpr double scale_factor_hold_bmi{ 1. };
    constexpr bool adapt_learning_rate_bmi{ false };
    constexpr bool adapt_learning_rate_manual{ false };

    constexpr double lr_bmi{ 2.e-5 };     // was 0.5e-5
    double lr1[4] = {lr, lr, 0, lr};
    double lr2[4] = {lr_bmi, lr_bmi, 0, lr_bmi};
    TrainParamsRNN p_manual(nb_manual_epochs, lr1, verbose, nb_manual_epochs,
                                   learn_after_each_ex, scale_factor_hold_manual, adapt_learning_rate_manual, learning_rate_adaptation);
    TrainParamsRNN p_bmi_no_CLDA(nb_bmi_epochs, lr2, verbose, nb_bmi_epochs,
                                 learn_after_each_ex, scale_factor_hold_bmi, adapt_learning_rate_bmi, learning_rate_adaptation);

    // CLDA variables and parameters
    int next_CLDA_day = 1;
    constexpr int CLDA_interval = 1; // days
    constexpr int nb_CLDA_steps{ 1 };
    const int nb_epochs_per_CLDA_steps{ nb_bmi_epochs/nb_CLDA_steps };
    constexpr double stopping_CLDA_criterion{ 0. };  //was -2

    TrainParamsRNN p_bmi_CLDA(nb_epochs_per_CLDA_steps, lr2, verbose, nb_epochs_per_CLDA_steps,
                                   learn_after_each_ex, scale_factor_hold_bmi, adapt_learning_rate_bmi, learning_rate_adaptation);
    TrainParamsRNN p_bmi_no_verbose_CLDA(nb_epochs_per_CLDA_steps, lr2, false, nb_epochs_per_CLDA_steps,
                                   learn_after_each_ex, scale_factor_hold_bmi, adapt_learning_rate_bmi, learning_rate_adaptation);
    
    // Unit-swapping parameters
    constexpr int unit_swap_day{ 10 };
    constexpr int nb_swap_units{ 0 };
    
    // Saving parameters
    constexpr bool save_CLDA_intermediate_steps{ false };
    constexpr bool save_CLDA_data{ false };
    constexpr int interval_CLDA_save{ 2 };
    
    // Dropout
    constexpr bool dropout{ false };
    constexpr bool dropout_during_clda{ false };
    constexpr double dropout_prob{ 0.1 };
    
    // Compactness
    constexpr int n_reals_compact{ 25 }; // nb of realizations of loss to evaluate mean and std on
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //                                  Simulation                                   //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Pre-train input FFN
    TorqueBasedArm arm(2, 0.04, radius_for_reset);
    constexpr double max_normalized_distance{ 5. };
    MixedGenerator tg(dim, max_normalized_distance);

    const unsigned int input_dim = tg.get_input_dim();
    const unsigned int output_dim = tg.get_output_dim();
    const int ffn_first_hidden_layer_size{ n_rec };
    FFN ffn({input_dim, n_rec, output_dim}, phi);
    if (ffn_is_pretrained) {
        ffn.load("../data/arm_model/seed"+to_string(seed_matrix)+"/ffn");
        ffn.set_input_type(tg.get_datatype());
    }
    else {
        ffn.train(tg, 1000, 1e-4, 1e-4, 0.999, 100); // train for 100 epochs, lr_hidden = 2e-4, lr_output = 2e-4, Adam beta2 = 0.999, batch_size = 100
    }
    ffn.save(dir+"ffn");
    Eigen::MatrixXd U = ffn.get_U();
    
    // Effector, inputs and targets
    FFNInput inputs(&arm, &ffn, n_targets, use_context_bit, duration, distance_to_target, hold_duration, feedback_delay);
    Target targets(dim, n_targets, distance_to_target);
    
    // Network
    const int n_in{ inputs.input_size() };
    RNN net(n_in, n_rec, n_out, noise_amplitude, tau_m, alpha_reward, phi);
    if (rnn_is_pretrained) net.load("../data/arm_model/seed"+to_string(seed_matrix)+"/manual_trained_network");
    else net.setU(U);
    
    // Training objectives
    Loss * loss_manual = new EndLoss(gamma_v_manual, gamma_f, effort_penalty_manual, activity_regularizer);
    Loss * loss_BMI = new EndLoss(gamma_v_bmi, 0., effort_penalty_bmi, activity_regularizer);
    //Loss * loss_manual = new CumulativeEndLoss(50, effort_penalty_manual, 0.01, distance_to_target, activity_regularizer);
    //Loss * loss_BMI = new CumulativeEndLoss(25, effort_penalty_bmi, 0.01, distance_to_target, activity_regularizer);
    
    
    if (not rnn_is_pretrained) {
        net.save(dir+"naive_network");
        save_to_file(dir, "naive", net, &arm, inputs, n_reals);
    }
    
    //=================================================================================//
    //                  PART 1: Pretraining under manual control                       //
    //=================================================================================//
    if (not rnn_is_pretrained){
        // Training under manual control
        cout << "Training under manual control..." << endl;
        
        net.train(loss_manual, &arm, inputs, targets, p_manual_before, dir+"manual_trained_network_", ofstream::out, OptimizerType::Adam);
        
        cout << "End manual training" << endl << endl;
    }
    // Save data for initial manual control
    net.save(dir+"manual_trained_network");
    save_to_file(dir, "manual_before_learning", net, &arm, inputs, 5);
    
    
    //=================================================================================//
    //                      PART 2: Fitting Kalman filter                              //
    //=================================================================================//

    // Gather data
    Monitor m_neuron, m_arm;
    gather_data(net, &arm, m_neuron, m_arm, inputs, targets, n_reals);

    // Fit decoder
    Readout readout(n_readouts, n_rec);
    VelocityKalmanFilter kf(readout, 2, radius_for_reset);
    kf.encode(m_arm.get_data(), m_neuron.get_data(), duration + hold_duration, 0., ReadoutSelectionMethod::random);
    
    // Save data before learning under BMI control
    inputs.set_context(&kf);
    save_to_file(dir, "bci_before_learning", net, &kf, inputs, n_reals);
    kf.save(dir);
    
    // Save valid readout units
    vector<size_t> readout_unit_ids = kf.get_readout_unit_ids();
    ofstream f_valid_units(dir+"valid_units_pre_swap.txt");
    for (auto u : readout_unit_ids) f_valid_units << u << '\n';
    f_valid_units.close();

    //=================================================================================//
    //             PART 3: Continual learning of manual and BMI tasks                  //
    //=================================================================================//
    int n{ 1 };
    double reward{net.compute_median_reward(loss_BMI, &kf, inputs, targets, p_bmi_no_CLDA, 10)};
    inputs.set_context(&arm);
    double reward_manual{net.compute_median_reward(loss_manual, &arm, inputs, targets, p_manual, 10)};
    
    int n_CLDA_periods{ 0 };
    
    int nb_day_with_good_perf{ 0 };
    
    while (n <= nb_days){
        
        inputs.set_context(&kf);
        
        if (n == next_CLDA_day and reward < stopping_CLDA_criterion){
            next_CLDA_day += CLDA_interval;
            //cout << "Performing CLDA on day " << n << endl;
            net.init_long_reward_traces(n_targets);
            n_CLDA_periods++;
            
            for (int clda_step(0); clda_step<nb_CLDA_steps; clda_step++){
                //Save pre-CLDA data
                if (save_CLDA_intermediate_steps){
                    save_to_file(dir, "bci_pre_clda_iteration"+to_string(n)+"_clda_step"+to_string(clda_step), net, &kf, inputs);
                }
                // Perform CLDA
                gather_data(net, &kf, m_neuron, m_arm, inputs, targets, n_reals);
                Eigen::MatrixXd bci_activity = kf.get_readout().read(m_neuron.get_data());
                Eigen::MatrixXd vel = m_arm.get_data();
                kf.clda(vel, bci_activity, targets, duration, hold_duration, alpha_CLDA, alpha_CLDA);
                
                //Save post-CLDA data
                if (save_CLDA_intermediate_steps){
                    save_to_file(dir, "bci_post_clda_pre_training_iteration"+to_string(n)+"_clda_step"+to_string(clda_step), net, &kf, inputs);
                }
                // Train
                if (n==1 and clda_step == 0) {
                    if(dropout_during_clda) net.train_with_dropout(loss_BMI, kf, inputs, targets, p_bmi_CLDA, dir+"bmi_", ofstream::out, OptimizerType::Adam);
                    else net.train(loss_BMI, &kf, inputs, targets, p_bmi_CLDA, dir+"bmi_", ofstream::out, OptimizerType::Adam);
                } else {
                    if(dropout_during_clda) net.train_with_dropout(loss_BMI, kf, inputs, targets, p_bmi_no_verbose_CLDA, dir+"bmi_", ofstream::out | ofstream::app, OptimizerType::Adam);
                    else net.train(loss_BMI, &kf, inputs, targets, p_bmi_no_verbose_CLDA, dir+"bmi_", ofstream::out | ofstream::app, OptimizerType::Adam);
                }
                //Save post-training data
                if (save_CLDA_intermediate_steps){
                    save_to_file(dir, "bci_post_clda_post_training_iteration"+to_string(n)+"_clda_step"+to_string(clda_step), net, &kf, inputs);
                }
                
            }
            if (n_CLDA_periods == 1){ // compute compactness for "early" training
                evaluate_compactness(dir, "early_", n_readouts, net, loss_BMI, kf, inputs, targets, p_bmi_no_CLDA);
                inputs.set_context(&kf);
            }
            net.reset_long_reward_traces();
        } else {
            if (n == 1) {
                if(dropout) net.train_with_dropout(loss_BMI, kf, inputs, targets, p_bmi_no_CLDA, dir+"bmi_", ofstream::out, OptimizerType::Adam);
                else net.train(loss_BMI, &kf, inputs, targets, p_bmi_no_CLDA, dir+"bmi_", ofstream::out, OptimizerType::Adam);
            }
            else {
                if(dropout) net.train_with_dropout(loss_BMI, kf, inputs, targets, p_bmi_no_CLDA, dir+"bmi_", ofstream::out | ofstream::app, OptimizerType::Adam);
                else net.train(loss_BMI, &kf, inputs, targets, p_bmi_no_CLDA, dir+"bmi_", ofstream::out | ofstream::app, OptimizerType::Adam);
            }
        }
        
        // Recording activities, trajectories and weights
        if (save_CLDA_data and (1+n)%interval_CLDA_save==0){
            net.save(dir+"network_day"+to_string(n));
            save_to_file(dir, "bci_day"+to_string(n), net, &kf, inputs, 5);
        }
        n++;
        reward = net.compute_median_reward(loss_BMI, &kf, inputs, targets, p_bmi_no_CLDA, 10);
        inputs.set_context(&arm);
        reward_manual = net.compute_median_reward(loss_manual, &arm, inputs, targets, p_manual, 10);
        //if (reward >= -1 and reward_manual >= -1) nb_day_with_good_perf++;
        if (reward >= -1) nb_day_with_good_perf++;
        //cout << '\n';
    }
    ofstream f(dir+"number_of_training_days.txt");
    f << n-1;
    f.close();
    
    // Save to file after learning
    kf.save(dir+"post_learning_");
    net.save(dir+"bci_trained_network");
    inputs.set_context(&kf);
    save_to_file(dir, "bci", net, &kf, inputs, n_reals);
    save_to_file(dir, "manual_with_bci_context", net, &arm, inputs, n_reals);
    
    inputs.set_context(&arm);
    save_to_file(dir, "manual", net, &arm, inputs, n_reals);
    
    
    //=================================================================================//
    //                          PART 4: Evaluate compactness                           //
    //=================================================================================//
    evaluate_compactness(dir, "late_", n_readouts, net, loss_BMI, kf, inputs, targets, p_bmi_no_CLDA);
    
    
    //=================================================================================//
    //                          PART 5: Generalization                                 //
    //=================================================================================//
    FFNInput inputs_gen(&arm, &ffn, 16, use_context_bit, duration, distance_to_target, hold_duration);
    Target targets_gen(dim, 16, distance_to_target);

    inputs_gen.set_context(&kf);
    
    double reward_generalization =  net.compute_median_reward(loss_BMI, &kf, inputs_gen, targets_gen, p_bmi_no_CLDA, 10);
    ofstream f_gen(dir+"reward_generalization.txt");
    f_gen << reward_generalization;
    f_gen.close();
    
    save_to_file(dir, "gen_bci", net, &kf, inputs_gen, n_reals);
    //save_to_file(dir, "gen_manual_with_bci_context", net, &arm, inputs_gen, 1);
    
    //inputs_gen.set_context(&arm);
    //save_to_file(dir, "gen_manual", net, &arm, inputs_gen, 1);
    
    
    // Delete
    delete loss_manual;
    delete loss_BMI;
    
    return 0;
}



// TODO: probably better to put a member function of RNN?
void evaluate_compactness(const string& dir, const string &file_suffix, int n_readouts, RNN &net, Loss * loss, const VelocityKalmanFilter &kf, Input & in, Target & target, TrainParamsRNN & p, int n_reals){
    // 1) Unit dropping curve
    vector<double> rewards(n_readouts);
    
    // Compute average reward with all readout units, to establish a reference
    VelocityKalmanFilter kf_ndc{ kf };
    in.set_context(&kf_ndc);
    rewards[0] = net.compute_median_reward(loss, &kf_ndc, in, target, p, n_reals);
    
    // Compute average reward after dropping the least dominant unit in succession
    for (int n_unit_removed{ 1 }; n_unit_removed<n_readouts; n_unit_removed++){
        vector<VelocityKalmanFilter> kfs_to_test(n_readouts - n_unit_removed + 1, kf_ndc);

        vector<double> test_rewards(n_readouts - n_unit_removed + 1);
        vector<size_t> readout_unit_ids = kf_ndc.get_readout_unit_ids();
        for (size_t id=0; id<n_readouts - n_unit_removed + 1; id++){
            kfs_to_test[id].remove_unit(readout_unit_ids[id]);
            in.set_context(&kfs_to_test[id]); //TODO: output error message when context is not changed; it should already
            test_rewards[id] = net.compute_median_reward(loss, &kfs_to_test[id], in, target, p, n_reals);
        }
        vector<size_t> sorted_rewards_id = argsort(test_rewards);  // sort indices to max reward, i.e. the unit removal yielding the less effect on the loss
        rewards[n_unit_removed] = test_rewards[sorted_rewards_id.back()];
        
        kf_ndc.remove_unit(readout_unit_ids[sorted_rewards_id.back()]);
        
        in.set_context(&kf_ndc);
        if ((n_unit_removed-1)%5==0) save_to_file(dir, "bci_" + file_suffix + "nb_units_removed_"+to_string(n_unit_removed), net, &kf_ndc, in, 5);
    }
    // Output to file
    ofstream outfile_compactness_ndc(dir+"compactness_"+file_suffix+"ndc.txt");
    for (auto u : rewards){
        outfile_compactness_ndc << u << endl;
    }
    outfile_compactness_ndc.close();
    
    
    // 2) Unit adding curve: adding the most important units, in order
    VelocityKalmanFilter kf_nac{ kf };
    in.set_context(&kf_nac);
    vector<size_t> units_added;
    vector<size_t> readout_ids = kf_nac.get_readout_unit_ids();
    for (int n_unit_added{ 1 }; n_unit_added<=n_readouts; n_unit_added++){
        vector<VelocityKalmanFilter> kfs_to_test(n_readouts - n_unit_added + 1, kf_nac);
        vector<double> test_rewards(n_readouts - n_unit_added + 1);

        // Add each unit not already added in turn and evaluate loss
        int test_count{ 0 };
        vector<size_t> tested_ids;
        for (auto u: readout_ids){
            vector<size_t> test_subset = units_added;
            if ( not contain(units_added, u) ) {
                test_subset.push_back(u);
                tested_ids.push_back(u);
                kfs_to_test[test_count].only_include_units(test_subset);
                in.set_context(&kfs_to_test[test_count]);
                test_rewards[test_count] = net.compute_median_reward(loss, &kfs_to_test[test_count], in, target, p, n_reals);
                test_count++;
            }
        }
        
        vector<size_t> sorted_rewards_id = argsort(test_rewards);
        rewards[n_unit_added-1] = test_rewards[sorted_rewards_id.back()];  // select the unit that most affects reward (rewards are negative)
        units_added.push_back(tested_ids[sorted_rewards_id.back()]);
        
        in.set_context(&kfs_to_test[sorted_rewards_id.back()]);
        if ((n_unit_added-1)%5==0) save_to_file(dir, "bci_" + file_suffix + "nb_units_added_"+to_string(n_unit_added), net, &kfs_to_test[sorted_rewards_id.back()], in, 5);
    }
    // Output to file
    ofstream outfile_compactness_nac(dir+"compactness_"+file_suffix+"nac.txt");
    for (auto u : rewards){
        outfile_compactness_nac << u << endl;
    }
    outfile_compactness_nac.close();
    
    
    // 3) Synergies (compute loss for all combinations of units)
    ofstream outfile_synergy_combination(dir+"synergy_combinations_"+file_suffix+".txt");
    ofstream outfile_synergy(dir+"synergy_"+file_suffix+".txt");
    for (int k=1; k<=n_readouts; ++k){
        VelocityKalmanFilter kf_syn{ kf };
        vector< vector<size_t> > combinations = find_combinations(kf_syn.get_readout_unit_ids(), k);
        vector<size_t> readout_units_ids = kf_syn.get_readout_unit_ids();
        for (int c=0; c<combinations.size(); ++c){
            VelocityKalmanFilter kf_loc{ kf };
            kf_loc.only_include_units(combinations[c]);
            in.set_context(&kf_loc);
            double reward = net.compute_mean_reward(loss, &kf_loc, in, target, p, 1);
            
            outfile_synergy << k << '\t' << reward << '\n';
            
            for (int i{0}; i<readout_units_ids.size(); ++i){
                if (contain(combinations[c], readout_units_ids[i])) outfile_synergy_combination << 1 << '\t';
                else  outfile_synergy_combination << 0 << '\t';
            }
            outfile_synergy_combination << reward << '\n';
        }
    }
    outfile_synergy.close();
    outfile_synergy_combination.close();
}

