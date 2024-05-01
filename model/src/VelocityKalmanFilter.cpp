/**
 * VelocityKalmanFilter.cpp | bmi_model
 */
#include "VelocityKalmanFilter.hpp"
#include <fstream>
#include "utilities.hpp"

VelocityKalmanFilter::VelocityKalmanFilter(Readout & _readout, int dim,  double radius_for_reset): Effector(dim, radius_for_reset), is_trained(false), readout(_readout) {
    effector_type = DECODER;

    v_hat = Vector::Zero(dim);
    P = Matrix::Zero(dim, dim);
    
    A = Matrix::Zero(dim, dim);
    W = Matrix::Zero(dim, dim);
    H = Matrix::Zero(readout.get_nb_channels(), dim);
    Q = Matrix::Zero(readout.get_nb_channels(), readout.get_nb_channels());
    m = Vector::Zero(readout.get_nb_channels());
}

void VelocityKalmanFilter::reset(){
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

void VelocityKalmanFilter::encode(const Matrix& velocities, const Matrix& unit_recordings, const int total_trial_duration, const int burnin, ReadoutSelectionMethod selection_method){
    assert(velocities.cols() == unit_recordings.cols());
    const int nb_time_steps{ int(velocities.cols()) };
    const int nb_trials{ nb_time_steps/total_trial_duration };

    Matrix X(unit_recordings.rows(), nb_time_steps - burnin * nb_trials);
    Matrix V(_dim, nb_time_steps - burnin * nb_trials);
    
    // Remove burnin period for each trial
    int post_burnin_count{ 0 };
    for (int trial_i=0; trial_i<nb_trials; trial_i++){
        for (int t=burnin; t<total_trial_duration; t++){
            X.col(post_burnin_count) = unit_recordings.col(trial_i*total_trial_duration + t);
            V.col(post_burnin_count) = velocities.col(trial_i*total_trial_duration + t);
            post_burnin_count++;
        }
    }
    
    // Select readout units according to activity
    if (not is_trained and check_valid){
        //Select valid units
        constexpr double rate_threshold{ 1. }; // ~ 1 spike for at least one of the targets during the trial; was 0.15/(double(trial_duration - burnin)*dt)
        const double std_threshold{ rate_threshold };
        int accumulated_readout_units{ 0 };

        Vector mean_activity = X.rowwise().mean();
        Vector std_activity = Eigen::sqrt(variance(X).array());
        
        if (selection_method == ReadoutSelectionMethod::best_of_all){
            std::vector<double> std_copy(std_activity.size());
            for (int r=0;r<std_activity.size();r++) std_copy[r] = std_activity(r);
            std::vector<size_t> sorted_stds_id = argsort(std_copy);
            
            for (std::vector<size_t>::reverse_iterator i=sorted_stds_id.rbegin();i!=sorted_stds_id.rend();++i){
                if (mean_activity(*i) > rate_threshold and accumulated_readout_units < readout.get_nb_channels()){
                    readout_unit_ids.push_back(*i);
                    accumulated_readout_units++;
                }
            }
        }
        else if (selection_method == ReadoutSelectionMethod::masking){
            constexpr float fraction_of_units_recorded{ 0.30 };
            const int nb_recorded_units{int(fraction_of_units_recorded*std_activity.size())};
            std::vector<double> std_copy(nb_recorded_units);
            for (int r=0;r<nb_recorded_units;r++) std_copy[r] = std_activity(r);
            std::vector<size_t> sorted_stds_id = argsort(std_copy);
            
            for (std::vector<size_t>::reverse_iterator i=sorted_stds_id.rbegin();i!=sorted_stds_id.rend();++i){
                if (mean_activity(*i) > rate_threshold and accumulated_readout_units < readout.get_nb_channels()){
                    readout_unit_ids.push_back(*i);
                    accumulated_readout_units++;
                }
            }
        }
        else if (selection_method == ReadoutSelectionMethod::random){
            for (size_t i=0;i<mean_activity.size();++i){
                if (mean_activity(i) > rate_threshold and accumulated_readout_units < readout.get_nb_channels()){
                    readout_unit_ids.push_back(i);
                    accumulated_readout_units++;
                }
            }
        }
        if (accumulated_readout_units != readout.get_nb_channels()){
            std::cout << "Note: Number of readout units found ("<<accumulated_readout_units<<"), is smaller than the number demanded ("<<readout.get_nb_channels()<<")\n";
        }
            
        //assert(accumulated_readout_units == readout.get_nb_channels());
        readout.set_readout_matrix(readout_unit_ids);
        
        //Resize H and Q matrix if the number of valid readout units is smaller than the original number
        H.resize(readout.get_nb_channels(), _dim);
        Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    }
    
    Matrix valid_recordings = readout.read(X);
    m = valid_recordings.rowwise().mean();
    valid_recordings.colwise() -= m;
    Matrix Ztranspose = valid_recordings.transpose();
    
    // Construct auxiliary matrices V1 and V2, such that V2 = V1 * A.transpose()
    Matrix V1(nb_time_steps-nb_trials-burnin*nb_trials, _dim), V2(nb_time_steps-nb_trials-burnin*nb_trials, _dim);
    Matrix Vtranspose = V.transpose();
    
    int row_counter=0;
    int trial_length_wo_burnin{ total_trial_duration - burnin };
    for (int trial_id=0;trial_id<nb_trials;trial_id++){
        for (int r=0; r<trial_length_wo_burnin-1; r++){
            V1.row(row_counter) = Vtranspose.row(trial_id*trial_length_wo_burnin + r);
            if (trial_id*trial_length_wo_burnin + r + 1 < V2.rows()) {
                V2.row(row_counter) = Vtranspose.row(trial_id*trial_length_wo_burnin + r+1);
            }
            row_counter++;
        }
    }
    
    // Find A by solving the least square (LS) problem: V1 * A.transpose() = V2
    // colPivHouseholderQr seems to be a sweet spot for fast and stable LS solver.
    Matrix Atranspose = V1.colPivHouseholderQr().solve(V2);
    A = Atranspose.transpose();
    
    // Find H by solving the LS problem: velocities * H.transpose() = Ztranspose
    Matrix Htranspose = Vtranspose.colPivHouseholderQr().solve(Ztranspose);
    H = Htranspose.transpose();
    
    // Covariance matrices
    W = (V2 - V1 * Atranspose).transpose() * (V2 - V1 * Atranspose) / (V1.rows() - 1);
    Q = (Ztranspose - Vtranspose * Htranspose).transpose() * (Ztranspose - Vtranspose * Htranspose) / nb_time_steps;
    
    is_trained = true;
}

Matrix VelocityKalmanFilter::decode(const Matrix& network_activity){
    
    long t_max = network_activity.cols();
    Matrix decoded_velocities(_dim, t_max);
    Matrix recordings = readout.read(network_activity);
    
    reset();
    if (is_trained){
        for (int t=0;t<t_max;t++){
            Vector v_hat_minus = A * v_hat;
            Matrix P_minus = A * P * A.transpose() + W;
            Matrix K = gain(P_minus);
            v_hat = v_hat_minus + K * (recordings.col(t) - m - H * v_hat_minus);
            P = (Matrix::Identity(_dim, _dim) - K * H) * P_minus;
            decoded_velocities.col(t) = v_hat;
        }
    } else {
        std::cerr << "Kalman filter not trained. Exiting." << std::endl;
        exit(EXIT_FAILURE);  //TODO: do not use exit!!!
    }
    return decoded_velocities;
}

void VelocityKalmanFilter::next_state(const Vector& network_activity){
    Vector recording = readout.read(network_activity);
    
    for (int d(0);d<_dim;++d){ state(d) = state(d) + dt * v_hat(d); }
    
    if (is_trained){
        Vector v_hat_minus = A * v_hat;
        Matrix P_minus = A * P * A.transpose() + W;
        Matrix K = gain(P_minus);
        v_hat = v_hat_minus + K * (recording - m - H * v_hat_minus);
        P = (Matrix::Identity(_dim, _dim) - K * H) * P_minus;
    } else {
        std::cerr << "Kalman filter not trained. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (int d(0);d<_dim;++d){ state(d + _dim) = v_hat(d); }
    end_effector = state;
}


Matrix VelocityKalmanFilter::gain(const Matrix& P_minus){
    Matrix T = H * P_minus * H.transpose() + Q;
    Eigen::ColPivHouseholderQR<Matrix> dec(T);
    if (dec.isInvertible()==false){
        std::cerr << "Noninvertible matrix. Exiting." << std::endl;
        std::cout << T << std::endl;
        exit(EXIT_FAILURE);
    }
    return P_minus * H.transpose() * dec.inverse();
}

Matrix VelocityKalmanFilter::get_end_effector_trajectory(const Matrix & unit_recordings){
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

void VelocityKalmanFilter::save(std::string file_prefix){
    std::ofstream f;
    
    // Readout units
    f.open(file_prefix + "kalman_readout_ids.txt");
    std::vector<size_t> readout_ids = get_readout_unit_ids();
    for (auto id : readout_ids){
        f << id << '\n';
    }
    f.close();
    
    // H matrix
    // TODO: beware of column-major default with Eigen
    f.open(file_prefix + "kalman_H.txt");
    for (int r(0);r<H.rows();r++) {
        for (int c(0);c<H.cols()-1;c++) {
            f <<H(r, c)<< ",";
        }
        f <<H(r, H.cols()-1)<< std::endl;
    }
    f.close();
    
    // A matrix
    f.open(file_prefix + "kalman_A.txt");
    for (int r(0);r<A.rows();r++) {
        for (int c(0);c<A.cols()-1;c++) {
            f <<A(r, c)<< ",";
        }
        f <<A(r, A.cols()-1)<< std::endl;
    }
    f.close();

    // W matrix
    f.open(file_prefix + "kalman_W.txt");
    for (int r(0);r<W.rows();r++) {
        for (int c(0);c<W.cols()-1;c++) {
            f <<W(r, c)<< ",";
        }
        f <<W(r, W.cols()-1)<< std::endl;
    }
    f.close();

    // Q matrix
    f.open(file_prefix + "kalman_Q.txt");
    for (int r(0);r<Q.rows();r++) {
        for (int c(0);c<Q.cols()-1;c++) {
            f <<Q(r, c)<< ",";
        }
        f <<Q(r, Q.cols()-1)<< std::endl;
    }
    f.close();

    // m vector
    f.open(file_prefix + "kalman_m.txt");
    for (int r(0);r<m.size();r++) {
        f << m(r) << '\n';
    }
    f.close();

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
    
    f.open(file_prefix + "kalman_gain.txt");
    for (int r(0);r<K.rows();r++) {
        for (int c(0);c<K.cols()-1;c++) {
            f <<K(r, c)<< ",";
        }
        f <<K(r, K.cols()-1)<< std::endl;
    }
    f.close();
}

void VelocityKalmanFilter::load(std::string file_prefix){
    std::ifstream file;
    std::string line;
    double val;
    int row_counter, col_counter;
    
    // Load readout ids
    readout_unit_ids.clear();
    int id{};
    file.open(file_prefix + "kalman_readout_ids.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file: " + file_prefix + "kalman_readout_ids.txt");
    while (std::getline(file, line)){
        std::stringstream ss(line);
        while (ss >> id) readout_unit_ids.push_back(id);
    }
    file.close();
    readout.set_readout_matrix(readout_unit_ids);
    
    // Resize matrices and vector
    H.resize(readout.get_nb_channels(), _dim);
    Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    A.resize(_dim, _dim);
    W.resize(_dim, _dim);
    m.resize(readout.get_nb_channels());

    //Loading H
    file.open(file_prefix + "kalman_H.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter = 0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            H(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading A
    file.open(file_prefix + "kalman_A.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            A(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading W
    file.open(file_prefix + "kalman_W.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            W(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading Q
    file.open(file_prefix + "kalman_Q.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            Q(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading m
    file.open(file_prefix + "kalman_m.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            m(col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
}



void VelocityKalmanFilter::clda(Matrix& velocities, const Matrix& unit_recordings, const std::vector<Vector> &targets, const int duration, const double alpha, const double beta){
    
    this->rotate_velocities(velocities, targets, duration);

    long nb_time_steps = velocities.cols();
    
    Vector m_clda = unit_recordings.rowwise().mean();
    Matrix Z = unit_recordings;
    for (int c(0);c<Z.cols();c++) Z.col(c) -= m_clda;
    Matrix Ztranspose = Z.transpose();
    Matrix Vtranspose = velocities.transpose();
    
    // Find H by solving the LS problem: velocities * H.transpose() = Ztranspose
    Matrix Htranspose = Vtranspose.colPivHouseholderQr().solve(Ztranspose);
    Matrix H_clda = Htranspose.transpose();
    
    // Covariance matrix
    Matrix Q_clda = (Ztranspose - Vtranspose * Htranspose).transpose() * (Ztranspose - Vtranspose * Htranspose) / nb_time_steps;
    
    // CLDA
    m = alpha * m + (1 - alpha) * m_clda;
    H = alpha * H + (1 - alpha) * H_clda;
    Q = beta * Q + (1 - beta) * Q_clda;
}

void VelocityKalmanFilter::clda(Matrix& velocities, Matrix& unit_recordings, Target& targets, const int duration, const int hold_duration, const double alpha, const double beta){
    
    long nb_time_steps = velocities.cols();
    int nb_targets_recorded = int(velocities.cols()) / (hold_duration + duration);

    // Rotate velocities
    this->rotate_velocities(velocities, targets, duration, hold_duration);

    /*
    // Remove time steps that might cause problems (infinite velocities) when rotating velocities  WHY???
    int nb_targets_recorded = int(velocities.cols()) / (hold_duration + duration);
    int time_steps_ignored{ 0 };  // DEBUG!!! was 3

    Matrix trimmed_velocities(velocities.rows(), nb_time_steps - nb_targets_recorded * time_steps_ignored);
    Matrix trimmed_unit_recordings(unit_recordings.rows(), nb_time_steps - nb_targets_recorded * time_steps_ignored);
    
    int column{ 0 };
    for (int trial=0; trial < nb_targets_recorded; trial++){
        for (int t=0; t<hold_duration + duration; t++) {
            int total = trial * (hold_duration + duration) + t;
            if (t > time_steps_ignored) {
                trimmed_velocities.col(column) = velocities.col(total);
                trimmed_unit_recordings.col(column) = unit_recordings.col(total);
                column++;
            }
        }
    }
    */
    // Performed CLDA
    Vector m_clda = unit_recordings.rowwise().mean();
    Matrix Z = unit_recordings;
    for (int c(0);c<Z.cols();c++) Z.col(c) -= m_clda;
    Matrix Ztranspose = Z.transpose();
    Matrix Vtranspose = velocities.transpose();
    
    // Find H by solving the LS problem: velocities * H.transpose() = Ztranspose
    Matrix Htranspose = Vtranspose.colPivHouseholderQr().solve(Ztranspose);
    Matrix H_clda = Htranspose.transpose();
    
    // Covariance matrix
    Matrix Q_clda = (Ztranspose - Vtranspose * Htranspose).transpose() * (Ztranspose - Vtranspose * Htranspose) / (nb_time_steps - nb_targets_recorded * hold_duration);
    
    // CLDA
    m = alpha * m + (1 - alpha) * m_clda;
    H = alpha * H + (1 - alpha) * H_clda;
    Q = beta * Q + (1 - beta) * Q_clda;
}

void VelocityKalmanFilter::rotate_velocities(Matrix& velocities, const std::vector<Vector> &targets, const int duration) {
    
    int number_of_targets = int(velocities.cols()) / duration;
    Vector position;

    int count(0);
    for (int t=0;t<number_of_targets;t++){
        position = Vector::Zero(_dim);
        for (int c=t*duration;c<(t+1)*duration;c++){
            //f << position.transpose() << std::endl;
            Vector tmp_vel = velocities.col(count);
            velocities.col(count) = (velocities.col(count).norm() / (targets[t % targets.size()]-position).norm()) * (targets[t % targets.size()]-position);
            position += tmp_vel*dt;
            count++;
        }
    }
}

void VelocityKalmanFilter::rotate_velocities(Matrix& velocities, Target& targets, const int duration, const int hold_duration) {
    int nb_targets{ targets.get_nb_targets() };
    int nb_targets_recorded{ int(velocities.cols()) / (hold_duration + duration) };
    Vector position(_dim);

    int count{ 0 };
    for (int d{ 0 };d<nb_targets_recorded;d++){
        Vector position(_dim);
        position << end_effector_init_cond(0), end_effector_init_cond(1);
        for (int c=d*(duration+hold_duration);c<(d+1)*(duration+hold_duration);c++){
            Vector tmp_vel = velocities.col(count);
            Vector tgt(_dim);
            if (c - d*(duration+hold_duration) < hold_duration) tgt = Vector::Zero(_dim);
            else tgt = targets(d % nb_targets);
            if ((tgt-position).norm() > 0) velocities.col(count) = (velocities.col(count).norm() / (tgt-position).norm()) * (tgt-position);
            else velocities.col(count) = Vector::Zero(_dim);
            position += tmp_vel*dt;
            count++;
        }
    }
}

void VelocityKalmanFilter::select_units_to_discard(const Matrix& bmi_unit_recordings, const int nb_units){
    Vector mean_activity = bmi_unit_recordings.rowwise().mean();
    //Vector std_activity = Eigen::sqrt(variance(bmi_unit_recordings).array());
    std::vector<double> m(mean_activity.size());
    swapped_out_units.clear();
    swapped_out_units.reserve(nb_units);
    
    for (int i=0; i<mean_activity.size(); i++) m[i] = mean_activity(i);
    std::vector<size_t> sorted_indices = argsort(m);
    
    std::cout << "Mean activity of discarded units: ";
    for (int i=0; i<nb_units; i++) {
        swapped_out_units.push_back(readout_unit_ids[sorted_indices[i]]);
        std::cout << m[sorted_indices[i]] << "\n";
    }
    std::cout << std::endl;
}

void VelocityKalmanFilter::swap_units(Matrix& velocities, const Matrix& all_unit_recordings, const int trial_duration, const int nb_units, const int burnin){
    const Vector mean_activity = all_unit_recordings.rowwise().mean();
    Vector var_activity = Vector::Zero(all_unit_recordings.rows());
    std::vector<double> s(var_activity.size(), 0.);
    
    // Select non-readout units to add
    const int nb_time_steps{ int(velocities.cols()) };
    const int nb_trials{ nb_time_steps/trial_duration };
    for (int trial{ 0 }; trial<nb_trials; trial++){
        const Matrix trial_resolved_activity = all_unit_recordings.block(0, trial*trial_duration, all_unit_recordings.rows(), trial_duration);
        for (int i=0; i<var_activity.size(); i++) var_activity += variance(trial_resolved_activity);
    }
    
    for (int i=0; i<mean_activity.size(); i++) s[i] = var_activity(i);
    std::vector<size_t> sorted_indices = argsort(s);
    
    int nb_added_units{ 0 };
    for (long i=sorted_indices.size()-1; i>=0; i--){
        if (mean_activity(sorted_indices[i]) >= 1. and nb_added_units < nb_units and not contain(readout_unit_ids, sorted_indices[i])) {
            readout_unit_ids.push_back(sorted_indices[i]);
            nb_added_units++;
        }
    }
    assert(nb_added_units == nb_units);
    const int old_size{ readout.get_nb_channels() };
    readout.set_readout_matrix(readout_unit_ids);
    
    // Save pre-swap KF parameters
    const Matrix A_tmp = A;
    const Matrix W_tmp = W;
    const Matrix H_tmp = H;
    const Matrix Q_tmp = Q;
    const Vector m_tmp = m;
    H.resize(readout.get_nb_channels(), _dim);
    Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    m.resize(readout.get_nb_channels());
    
    encode(velocities, all_unit_recordings, trial_duration, burnin);
    
    // Reassign A and W, as these won't change due to the swap
    A = A_tmp;
    W = W_tmp;
    
    // Rebuild m, H, and Q
    Matrix P = Matrix::Zero(old_size, readout.get_nb_channels());  // projection matrix onto new set of readout units
    std::vector<size_t> new_readout_units(old_size);
    int cum_sum_readout{ 0 };
    for (int c=0; c<P.cols(); c++){
        if (contain(swapped_out_units, readout_unit_ids[c]) == false) {
            P(cum_sum_readout, c) = 1.;
            new_readout_units[cum_sum_readout] = readout_unit_ids[c];
            cum_sum_readout++;
        }
    }
  
    readout_unit_ids.swap(new_readout_units);
    readout.set_readout_matrix(readout_unit_ids);
    const Vector m_new = P * m;
    const Matrix H_new = P * H;
    const Matrix Q_new = P * Q * P.transpose();
        
    H.resize(readout.get_nb_channels(), _dim);
    Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    m.resize(readout.get_nb_channels());
    
    m = m_new;
    H = H_new;
    Q = Q_new;
        
    m.head(old_size - nb_units) = P.topLeftCorner(P.rows()-nb_units, P.cols()-nb_units) * m_tmp;
    H.topRows(old_size - nb_units) = P.topLeftCorner(P.rows()-nb_units, P.cols()-nb_units) * H_tmp;
    Q.topLeftCorner(old_size - nb_units, old_size - nb_units) = P.topLeftCorner(P.rows()-nb_units, P.cols()-nb_units) * Q_tmp * P.topLeftCorner(P.rows()-nb_units, P.cols()-nb_units).transpose();
}

void VelocityKalmanFilter::remove_unit(const size_t id){
    assert(contain(readout_unit_ids, id));
    
    // Modify H, Q, m
    Matrix H_old = H;
    Matrix Q_old = Q;
    Vector m_old = m;
    
    Matrix proj = Matrix::Zero(readout.get_nb_channels()-1, readout.get_nb_channels());  // projection matrix onto new set of readout units
    std::vector<size_t> new_readout_units(readout.get_nb_channels()-1);
    int cum_sum_readout{ 0 };
    for (size_t c=0; c<proj.cols(); c++){
        if (readout_unit_ids[c] != id) {
            proj(cum_sum_readout, c) = 1.;
            new_readout_units[cum_sum_readout] = readout_unit_ids[c];
            cum_sum_readout++;
        }
    }
    readout_unit_ids.swap(new_readout_units);
    readout.set_readout_matrix(readout_unit_ids);

    const Vector m_new = proj * m;
    const Matrix H_new = proj * H;
    const Matrix Q_new = proj * Q * proj.transpose();
        
    H.resize(readout.get_nb_channels(), _dim);
    Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    m.resize(readout.get_nb_channels());
    
    m = m_new;
    H = H_new;
    Q = Q_new;
}

void VelocityKalmanFilter::remove_units(const std::vector<size_t>& ids){
    assert(ids.size() < readout_unit_ids.size());
    
    // Modify H, Q, m
    Matrix H_old = H;
    Matrix Q_old = Q;
    Vector m_old = m;
    
    Matrix proj = Matrix::Zero(readout.get_nb_channels()-ids.size(), readout.get_nb_channels());  // projection matrix onto new set of readout units
    std::vector<size_t> new_readout_units(readout.get_nb_channels()-ids.size());
    int cum_sum_readout{ 0 };
    for (size_t c=0; c<proj.cols(); c++){
        if (not contain(ids, readout_unit_ids[c])) {
            proj(cum_sum_readout, c) = 1.;
            new_readout_units[cum_sum_readout] = readout_unit_ids[c];
            cum_sum_readout++;
        }
    }
    readout_unit_ids.swap(new_readout_units);
    readout.set_readout_matrix(readout_unit_ids);

    const Vector m_new = proj * m;
    const Matrix H_new = proj * H;
    const Matrix Q_new = proj * Q * proj.transpose();
        
    H.resize(readout.get_nb_channels(), _dim);
    Q.resize(readout.get_nb_channels(), readout.get_nb_channels());
    m.resize(readout.get_nb_channels());
    
    m = m_new;
    H = H_new;
    Q = Q_new;
}

void VelocityKalmanFilter::only_include_units(const std::vector<size_t>& ids){
    assert(ids.size() <= readout_unit_ids.size());
    std::vector<size_t> units_to_remove;
    
    for (auto u : readout_unit_ids){
        if ( not contain(ids, u) ) units_to_remove.push_back(u);
    }
    
    remove_units(units_to_remove);
}
