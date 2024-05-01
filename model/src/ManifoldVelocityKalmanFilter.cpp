/**
 * ManifoldVelocityKalmanFilter.cpp | bmi_model
 */
#include "ManifoldVelocityKalmanFilter.hpp"
#include "utilities.hpp"
#include <fstream>

ManifoldVelocityKalmanFilter::ManifoldVelocityKalmanFilter(Readout &readout, FactorAnalysis &fa, int dim, double radius_for_reset): Effector(dim, radius_for_reset), is_trained(false), readout(readout), factor_analyser(fa) {
    effector_type = DECODER;
    
    v_hat = Vector::Zero(dim);
    P = Matrix::Zero(dim, dim);
    
    A = Matrix::Zero(dim, dim);
    W = Matrix::Zero(dim, dim);
    H = Matrix::Zero(fa.get_nb_factors(), dim);
    Q = Matrix::Zero(fa.get_nb_factors(), fa.get_nb_factors());
    K_ss = Matrix::Zero(dim, fa.get_nb_factors());
    
    readout_unit_ids.clear();
}

void ManifoldVelocityKalmanFilter::reset(){
    v_hat.setZero();
    P.setZero();
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

void ManifoldVelocityKalmanFilter::encode(const Matrix& velocities, const Matrix& unit_recordings, const int trial_duration, const bool is_cursor_passive){
    assert(velocities.cols() == unit_recordings.cols());
    int nb_time_steps = int(velocities.cols());
    const int nb_trials = nb_time_steps / trial_duration;
    const int burnin{ int(0.05*trial_duration) };   // discarded burnin/relaxation period TODO: put this as a function parameter
    
    // Remove burnin period for each trial
    Matrix X(unit_recordings.rows(), nb_time_steps - burnin * nb_trials);
    Matrix V(_dim, nb_time_steps - burnin * nb_trials);
    int post_burnin_count{ 0 };
    for (int trial_i=0; trial_i<nb_trials; trial_i++){
        for (int t=burnin; t<trial_duration; t++){
            X.col(post_burnin_count) = unit_recordings.col(trial_i*trial_duration + t);
            V.col(post_burnin_count) = velocities.col(trial_i*trial_duration + t);
            post_burnin_count++;
        }
    }
    
    if (not is_trained){
        //Select valid units
        const double rate_threshold{ 0.15/(double(trial_duration - burnin)*dt) }; // ~ 1 spike for at least one of the targets during the trial
        const double std_threshold{ rate_threshold };
        
        Vector mean_activity = X.rowwise().mean();
        Vector std_activity = Eigen::sqrt(variance(X).array());
        
        for (size_t i=0;i<mean_activity.size();i++) {
            if (mean_activity(i) > rate_threshold and std_activity(i) > std_threshold) readout_unit_ids.push_back(i);
        }
        readout.set_readout_matrix(readout_unit_ids);
    }
    Matrix valid_recordings = readout.read(X);
    factor_analyser.cross_val(5, 5, 50, valid_recordings);
    
    
    // z-score activities
    m = valid_recordings.rowwise().mean();
    s = Eigen::sqrt(variance(valid_recordings).array());
    valid_recordings.colwise() -= m;
    valid_recordings.array().colwise() /= s.array();
    
    //Learn factor analysis model
    factor_analyser.learn(valid_recordings);
    Matrix f = factor_analyser.infer(valid_recordings);
    
    // Fit Kalman
    m_f = f.rowwise().mean();
    s_f = Eigen::sqrt(variance(f).array());
    Matrix Z = f;
    Z.colwise() -= m_f;
    Z.array().colwise() /= s_f.array();
    
    Matrix Ztranspose = Z.transpose();

    // Construct auxiliary matrices V1 and V2, such that V2 = V1 * A.transpose()
    Matrix V1(nb_time_steps-(burnin+1)*nb_trials, _dim), V2(nb_time_steps-(burnin+1)*nb_trials, _dim);
    Matrix Vtranspose = V.transpose();
    
    int row_counter=0;
    for (int trial_id=0;trial_id<nb_trials;trial_id++){
        for (int r=trial_id*(trial_duration-burnin);r<(trial_id+1)*(trial_duration-burnin)-1;r++) {
            V1.row(row_counter) = Vtranspose.row(r);
            if (r+1 < V2.rows()) V2.row(row_counter) = Vtranspose.row(r+1);
            row_counter++;
        }
    }
    
    // Find A by solving the least square (LS) problem: V1 * A.transpose() = V2
    // colPivHouseholderQr seems to be a sweet spot for fast and stable LS solver.
    Matrix Atranspose;
    //A = Matrix::Identity(_dim, _dim);  //DEBUG!!!!!
    if (is_cursor_passive) A = Matrix::Identity(_dim, _dim);
    else {
        Atranspose = V1.colPivHouseholderQr().solve(V2);
        A = Atranspose.transpose();
    }
    
    // Find H by solving the LS problem: velocities * H.transpose() = Ztranspose
    Matrix Htranspose = Vtranspose.colPivHouseholderQr().solve(Ztranspose);
    H = Htranspose.transpose();
    
    // Covariance matrices
    //W = 2.*Matrix::Identity(_dim, _dim); //DEBUG!!!!
    if (is_cursor_passive) W = 2.*Matrix::Identity(_dim, _dim);
    else W = (V2 - V1 * Atranspose).transpose() * (V2 - V1 * Atranspose) / (V1.rows() - 1);
    //std::cout << "W = \n";
    //std::cout << W << std::endl;
    Q = (Ztranspose - Vtranspose * Htranspose).transpose() * (Ztranspose - Vtranspose * Htranspose) / nb_time_steps;
    
    // Compute steady-state Kalman gain
    K_ss = steady_state_Kalman_gain();
    
    is_trained = true;
}

Matrix ManifoldVelocityKalmanFilter::decode(const Matrix& network_activity){
    
    long t_max = network_activity.cols();
    Matrix decoded_velocities(_dim, t_max);
    Matrix recordings = readout.read(network_activity);
    
    recordings.colwise() -= m;
    recordings.array().colwise() /= s.array();

    Matrix f = factor_analyser.infer(recordings);
    f.colwise() -= m_f;
    f.array().colwise() /= s_f.array();
    
    reset();
    if (is_trained){
        for (int t=0;t<t_max;t++){
            Vector v_hat_minus = A * v_hat;
            Matrix P_minus = A * P * A.transpose() + W;
            //Matrix K = gain(P_minus);
            v_hat = v_hat_minus + K_ss * (f.col(t) - H * v_hat_minus);
            P = (Matrix::Identity(_dim, _dim) - K_ss * H) * P_minus;
            decoded_velocities.col(t) = v_hat;
        }
    } else {
        std::cerr << "Kalman filter not trained. Exiting." << std::endl;
        exit(EXIT_FAILURE);  //TODO: do not use exit!!!
    }
    return decoded_velocities;
}

Vector ManifoldVelocityKalmanFilter::compute_openloop_velocity(const Matrix& network_activity){
    
    Vector predicted_velocity(_dim);
    Matrix recordings = readout.read(network_activity);
    
    recordings.colwise() -= m;
    recordings.array().colwise() /= s.array();

    Matrix f = factor_analyser.infer(recordings);
    f.colwise() -= m_f;
    f.array().colwise() /= s_f.array();
    
    Vector mean_f = f.rowwise().mean();
    
    reset();
    if (is_trained) predicted_velocity = K_ss * mean_f;
    else {
        std::cerr << "Kalman filter not trained. Exiting." << std::endl;
        exit(EXIT_FAILURE);  //TODO: do not use exit!!!
    }
    return predicted_velocity;
}


void ManifoldVelocityKalmanFilter::next_state(const Vector& network_activity){
    Vector recording = readout.read(network_activity);
    recording = (recording - m).array()/s.array();
    
    Vector f = (factor_analyser.infer(recording) - m_f).array()/s_f.array();
    
    for (int d(0);d<_dim;++d){ state(d) = state(d) + dt * v_hat(d); }
    
    if (is_trained){
        Vector v_hat_minus = A * v_hat;
        Matrix P_minus = A * P * A.transpose() + W;
        //Matrix K = gain(P_minus);
        v_hat = v_hat_minus + K_ss * (f - H * v_hat_minus);
        P = (Matrix::Identity(_dim, _dim) - K_ss * H) * P_minus;
    } else {
        std::cerr << "Kalman filter not trained. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (int d(0);d<_dim;++d){ state(d + _dim) = v_hat(d); }
    end_effector = state;
}


Matrix ManifoldVelocityKalmanFilter::gain(const Matrix& P_minus){
    Matrix T = H * P_minus * H.transpose() + Q;
    Eigen::ColPivHouseholderQR<Matrix> dec(T);
    if (dec.isInvertible()==false){
        std::cerr << "Noninvertible matrix. Exiting." << std::endl;
        std::cout << "H = \n";
        std::cout << H << std::endl;
        std::cout << "P_minus = \n";
        std::cout << P_minus << std::endl;
        std::cout << "Q = \n";
        std::cout << Q << std::endl;
        
        std::cout << "Noninvertible matrix:\n";
        std::cout << T << std::endl;
        exit(EXIT_FAILURE);
    }
    return P_minus * H.transpose() * dec.inverse();
}

Matrix ManifoldVelocityKalmanFilter::get_end_effector_trajectory(const Matrix & unit_recordings){
    reset();
    long total_time = unit_recordings.cols();
    Matrix s = Matrix::Zero(3*_dim, total_time);

    s.col(0) = end_effector;
    for (int t(1);t<total_time;t++){
        next_state(unit_recordings.col(t-1));
        s.col(t) = end_effector;
    }
    return s;
}

Matrix ManifoldVelocityKalmanFilter::steady_state_Kalman_gain(){
    
    // Iterative solution of the discrete-time algebraic Riccati equation for steady-state Kalman gain
    Matrix Sigma_minus = W;
    Matrix K = this->gain(Sigma_minus);
    Matrix K_prev;
    Matrix Sigma = (Matrix::Identity(_dim, _dim) - K * H) * Sigma_minus;
    
    do {
        K_prev = K;
        Sigma_minus = A * Sigma * A.transpose() + W;
        K = this->gain(Sigma_minus);
        Sigma = (Matrix::Identity(_dim, _dim) - K * H) * Sigma_minus;
    } while ((K - K_prev).norm() > 1e-6);
    
    
    return K;
}

Vector ManifoldVelocityKalmanFilter::compute_angles(const Matrix& vectors1, const Matrix& vectors2){
    Vector angles(vectors1.cols());
    
    for (int i=0; i<vectors1.cols(); i++){
        Vector v1 = vectors1.col(i);
        Vector v2 = vectors2.col(i);
        
        angles(i) = acos(v1.dot(v2) / v1.norm() / v2.norm());
    }
    return angles;
}

void ManifoldVelocityKalmanFilter::within_manifold_perturbation(const Matrix & unit_recordings, Loss * loss, Target & target, const double target_loss_per_target, const int trial_duration, const int hold_duration, const double scale_factor, const double tolerance){
        
    Matrix permut_mat;
    const int nb_targets{ target.get_nb_targets() };
    Vector target_loss_vec(nb_targets);
    Vector predicted_loss(nb_targets);
    
    const std::string method{ "openloop velocities" };
    //const std::string method{ "target loss" };

    target_loss_vec.setConstant(target_loss_per_target);
    predicted_loss.setZero();
    
    const double min_angle{ 20.*M_PI/180. };
    const double max_angle{ 75.*M_PI/180. };
    double min_ol_vel{ 0. }, max_ol_vel{ 0. };
    
    bool found_perturbation{ false };
    
    // Compute unperturbed predicted velocities
    Matrix unperturbed_velocities(_dim, nb_targets);
    if (method == "openloop velocities"){
        for (int i=0; i<nb_targets; i++) {
            Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
            unperturbed_velocities.col(i) = compute_openloop_velocity(recording_for_target_i);
            max_ol_vel += unperturbed_velocities.col(i).norm();
        }
        max_ol_vel /= double(nb_targets);
        min_ol_vel = 0.25*max_ol_vel;
        max_ol_vel *= 2.;
    }
    
    // Test perturbations
    Matrix perturbed_velocities(_dim, nb_targets);
    int loop_counter{ 0 };
    do {
        loop_counter ++;

        // Select perturbation
        permut_mat = get_permutation_matrix(factor_analyser.get_nb_factors());
        m_f = permut_mat * m_f;
        s_f = permut_mat * s_f;
        factor_analyser.set_beta(permut_mat * factor_analyser.get_beta());
        
        if (method == "openloop velocities"){
            // Compute perturbed predicted velocities
            for (int i=0; i<nb_targets; i++){
                Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
                perturbed_velocities.col(i) = compute_openloop_velocity(recording_for_target_i);
                //Matrix s = get_end_effector_trajectory(recording_for_target_i, initial_state);
                //predicted_loss(i) = (*loss)(s, target(i), hold_duration, scale_factor);
            }
            
            // Compute angles between target-wise unperturbed and perturbed velocities
            Vector angles = compute_angles(unperturbed_velocities, perturbed_velocities);
            found_perturbation = true;
            for (int i=0; i<nb_targets; i++){
                double norm_vel = perturbed_velocities.col(i).norm();
                if (angles(i) < min_angle || angles(i) > max_angle || norm_vel > max_ol_vel || norm_vel < min_ol_vel) {
                    found_perturbation = false;
                    break;
                }
            }
        }
        
        if (method == "target loss"){
            found_perturbation = true;
            for (int i=0; i<target.get_nb_targets(); i++){
                Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
                Matrix s = get_end_effector_trajectory(recording_for_target_i);  //DEBUG: should possible force non-random initial conditions
                double loc_loss = (*loss)(s, target(i), hold_duration, scale_factor);
                if (abs(loc_loss-target_loss_per_target) > tolerance) {
                    //std::cout << "Perturbed loss = " << loc_loss << '\n';
                    found_perturbation = false;
                    break;
                }
            }
        }
        if (loop_counter % 10000 == 0) std::cout << "Loop " << loop_counter << '\n';
    } while(not found_perturbation); //while ((predicted_loss - target_loss_vec).norm() > tolerance);
}


void ManifoldVelocityKalmanFilter::outside_manifold_perturbation(const Matrix & unit_recordings, Loss * loss, Target & target, const double target_loss_per_target, const int trial_duration, const int hold_duration, const double scale_factor, const double tolerance){
    
    Matrix permut_mat;
    const int nb_targets{ target.get_nb_targets() };
    Vector target_loss_vec(nb_targets);
    Vector predicted_loss(nb_targets);

    const std::string method{ "openloop velocities" };
    //const std::string method{ "target loss" };

    target_loss_vec.setConstant(target_loss_per_target);
    predicted_loss.setZero();

    const double min_angle{ 20.*M_PI/180. };
    const double max_angle{ 75.*M_PI/180. };
    double min_ol_vel{ 0. }, max_ol_vel{ 0. };

    bool found_perturbation{ false };

    // Compute unperturbed predicted velocities
    Matrix unperturbed_velocities(_dim, nb_targets);
    if (method == "openloop velocities"){
        for (int i=0; i<nb_targets; i++) {
            Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
            unperturbed_velocities.col(i) = compute_openloop_velocity(recording_for_target_i);
            max_ol_vel += unperturbed_velocities.col(i).norm();
        }
        max_ol_vel /= double(nb_targets);
        min_ol_vel = 0.25*max_ol_vel;
        max_ol_vel *= 2.;
    }

    // Test perturbations
    Matrix perturbed_velocities(_dim, nb_targets);
    int loop_counter{ 0 };
    do {
        loop_counter ++;

        // Select perturbation
        permut_mat = get_permutation_matrix(factor_analyser.get_p());
        factor_analyser.set_beta(factor_analyser.get_beta()*permut_mat);
        m = permut_mat * m;
        s = permut_mat * s;
        
        if (method == "openloop velocities"){
            // Compute perturbed predicted velocities
            for (int i=0; i<nb_targets; i++){
                Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
                perturbed_velocities.col(i) = compute_openloop_velocity(recording_for_target_i);
                //Matrix s = get_end_effector_trajectory(recording_for_target_i, initial_state);
                //predicted_loss(i) = (*loss)(s, target(i), hold_duration, scale_factor);
            }
            
            // Compute angles between target-wise unperturbed and perturbed velocities
            Vector angles = compute_angles(unperturbed_velocities, perturbed_velocities);
            found_perturbation = true;
            for (int i=0; i<nb_targets; i++){
                double norm_vel = perturbed_velocities.col(i).norm();
                if (angles(i) < min_angle || angles(i) > max_angle || norm_vel > max_ol_vel || norm_vel < min_ol_vel) {
                    found_perturbation = false;
                    break;
                }
            }
        }
        
        if (method == "target loss"){
            found_perturbation = true;
            for (int i=0; i<target.get_nb_targets(); i++){
                Matrix recording_for_target_i = unit_recordings.block(0, i*trial_duration, trial_duration, unit_recordings.rows());
                Matrix s = get_end_effector_trajectory(recording_for_target_i);
                double loc_loss = (*loss)(s, target(i), hold_duration, scale_factor);
                if (abs(loc_loss-target_loss_per_target) > tolerance) {
                    //std::cout << "Perturbed loss = " << loc_loss << '\n';
                    found_perturbation = false;
                    break;
                }
            }
        }
        if (loop_counter % 10000 == 0) std::cout << "Loop " << loop_counter << '\n';
    } while(not found_perturbation); //while ((predicted_loss - target_loss_vec).norm() > tolerance);
}




void ManifoldVelocityKalmanFilter::save(std::string file_prefix){
    std::ofstream f_gain(file_prefix + "kalman_gain.txt");
    std::ofstream f_H(file_prefix + "kalman_H.txt");
    std::ofstream f_A(file_prefix + "kalman_A.txt");
    
    output_eigen_matrix_to_file_stream(H, f_H); // H matrix
    output_eigen_matrix_to_file_stream(A, f_A); // A matrix (probably not useful, but whatever...)
    output_eigen_matrix_to_file_stream(K_ss, f_gain); // Steady-state Kalman gain
    
    f_gain.close();
    f_H.close();
    f_A.close();
}
