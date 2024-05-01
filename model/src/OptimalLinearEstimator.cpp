/**
 * OptimalLinearEstimator.cpp | bmi_model
 */
#include "OptimalLinearEstimator.hpp"
#include <random>

OptimalLinearEstimator::OptimalLinearEstimator(Readout & readout, int dim, OLEtype encoding_type, double radius_for_reset): Effector(dim, radius_for_reset), is_trained(false), readout(readout), encode_type(encoding_type), is_randomly_initialized(false), threshold_for_readout(0.5) {
    effector_type = DECODER;
    conversion_factor = 0.1;
    
    P = Matrix::Zero(readout.get_nb_channels(), dim);
    m = Vector::Zero(readout.get_nb_channels());
    b = Vector::Zero(readout.get_nb_channels());
    
    readout_unit_ids.clear();

}

void OptimalLinearEstimator::init_parameters(){
    std::mt19937 rng;
    std::uniform_real_distribution<double> unif_angle(0., 2*M_PI);
    m = Vector::Ones(readout.get_nb_channels());
    b = Vector::Ones(readout.get_nb_channels());
    for (int i(0);i<readout.get_nb_channels();i++){
        double angle = unif_angle(rng);
        if (_dim > 2) std::cout << "Dimension larger than 2 not supported" << std::endl;
        P(i, 0) = cos(angle);
        P(i, 1) = sin(angle);
    }
    is_randomly_initialized = true;
}

void OptimalLinearEstimator::reset(){
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

void OptimalLinearEstimator::encode(const Matrix& velocities_or_targets, const Matrix& network_activity, const int duration, const int hold_duration){
    Matrix readout_activity = readout.read(network_activity);
    if (encode_type == OLEtype::velocity) _encode_velocities(velocities_or_targets, readout_activity);
    else if(encode_type == OLEtype::target) _encode_targets(velocities_or_targets, readout_activity, duration, hold_duration);
    
    is_trained = true;
}

void OptimalLinearEstimator::_encode_velocities(const Matrix& velocities, const Matrix& unit_recordings){
    assert(velocities.cols() == unit_recordings.cols());  // nb of separate recordings
    int nb_time_steps = int(velocities.cols());
    
    Matrix V(nb_time_steps, 1+_dim);
    V.col(0) = Vector::Ones(nb_time_steps);
    for (int c=1;c<V.cols();c++){
        V.col(c) = velocities.row(c-1).transpose();
    }
    Matrix Y = unit_recordings.transpose();
    
    Matrix beta = V.colPivHouseholderQr().solve(Y);
    assert(beta.cols() == Y.cols());
    assert(beta.rows() == 1+_dim);
    
    b = beta.row(0).transpose();
    
    m = Vector::Zero(readout.get_nb_channels());
    
    for (int c=1; c<1+_dim; c++) {
        m.array() += beta.row(c).transpose().array() * beta.row(c).transpose().array();
    }
    m.array() = sqrt(m.array());
    
    if (is_randomly_initialized and not is_trained){
        //Select units according to whether modulation depth is large enough
        std::vector<size_t> readout_unit_ids;
        for (size_t i=0;i<m.size();i++) {
            if (m(i) > threshold_for_readout) readout_unit_ids.push_back(i);
        }
        readout.set_readout_matrix(readout_unit_ids);
        
        //Re-construct m and b using valid units
        Vector m_new(readout_unit_ids.size());
        Vector b_new(readout_unit_ids.size());
        P = Matrix::Zero(readout_unit_ids.size(), _dim);
        for (int i(0);i<readout_unit_ids.size();i++) m_new(i) = m(readout_unit_ids[i]);
        for (int i(0);i<readout_unit_ids.size();i++) b_new(i) = b(readout_unit_ids[i]);
        m = m_new;
        b = b_new;
        
        for (int r=0;r<P.rows();r++){
            for (int c=0; c<_dim; c++){
                P(r, c) = beta(c+1, readout_unit_ids[r]) / m(r);
            }
        }
    } else {
        for (int r=0;r<P.rows();r++){
            for (int c=0; c<_dim; c++){
                P(r, c) = beta(c+1, r) / m(r);
            }
        }
    }
    //std::cout << "Decoded preferred directions : \n";
    //std::cout << P << std::endl;
}

void OptimalLinearEstimator::_encode_targets(const Matrix& targets, const Matrix& unit_recordings, const int duration, const int hold_duration){
    const int nb_time_steps = int(unit_recordings.cols());
    const int nb_units = int(unit_recordings.rows());

    const int nb_trials = nb_time_steps / (duration + hold_duration);
    const int nb_targets = int(targets.rows()); //targets is nb_targets x dim
    const int dim = int(targets.cols());
    
    // Construct target matrix
    Matrix all_targets(nb_trials, 1+dim);
    for (int i=0; i<nb_trials; i++){
        all_targets(i, 0) = 1.;
        all_targets(i, 1) = targets(i % nb_targets, 0);
        all_targets(i, 2) = targets(i % nb_targets, 1);
    }
    
    //Compute mean rates (restricted to half of the "move" segment)
    Matrix mean_rates = Matrix::Zero(nb_trials, nb_units);
    for (int i=0; i<nb_trials; i++){
        for (int t=hold_duration; t<hold_duration+duration/2; t++) {
            mean_rates.row(i) += unit_recordings.col((duration+hold_duration)*i+t).transpose();
        }
        mean_rates.row(i) /= double(duration/2);
    }
    
    // Solve
    Matrix beta = all_targets.colPivHouseholderQr().solve(mean_rates);
    b = beta.row(0).transpose();
    m = Vector::Zero(readout.get_nb_channels());
    for (int c=1; c<1+_dim; c++) {
        m.array() += beta.row(c).transpose().array() * beta.row(c).transpose().array();
    }
    m.array() = sqrt(m.array());
    
    if (is_randomly_initialized and not is_trained){
        //Select units according to whether modulation depth is large enough
        for (size_t i=0;i<m.size();i++) {
            if (m(i) > threshold_for_readout) readout_unit_ids.push_back(i);
        }
        readout.set_readout_matrix(readout_unit_ids);
        
        //Re-construct m and b using valid units
        Vector m_new(readout_unit_ids.size());
        Vector b_new(readout_unit_ids.size());
        P = Matrix::Zero(readout_unit_ids.size(), _dim);
        for (int i(0);i<readout_unit_ids.size();i++) m_new(i) = m(readout_unit_ids[i]);
        for (int i(0);i<readout_unit_ids.size();i++) b_new(i) = b(readout_unit_ids[i]);
        m = m_new;
        b = b_new;
        
        for (int r=0;r<P.rows();r++){
            for (int c=0; c<_dim; c++){
                P(r, c) = beta(c+1, readout_unit_ids[r]) / m(r);
            }
        }
    } else {
        for (int r=0;r<P.rows();r++){
            for (int c=0; c<_dim; c++){
                P(r, c) = beta(c+1, r) / m(r);
            }
        }
    }
    //std::cout << "Decoded preferred directions : \n";
    //std::cout << P << std::endl;
}


void OptimalLinearEstimator::next_state(const Vector& network_activity){
    //if (is_trained){
        Vector recording = readout.read(network_activity);
        Vector z = (recording - b).array() / m.array();
        Vector v = P.colPivHouseholderQr().solve(z);
        if(encode_type == OLEtype::target) v *= conversion_factor;
        for (int d(0);d<_dim;++d) state(d) = state(d) + dt * state(d+_dim); //@todo: Chase et al. use a non-physical position(t) = position(t-1) + dt * velocity(t)
        for (int d(0);d<_dim;++d) state(d+_dim) = v(d);
        end_effector = state;
    //} else {
    //    std::cerr << "OLE filter not trained. Exiting." << std::endl;
    //    exit(EXIT_FAILURE);
    //}
}

Matrix OptimalLinearEstimator::get_end_effector_trajectory(const Matrix & unit_recordings){
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


void OptimalLinearEstimator::rotate_pds(double angle_in_degrees, double f){
    const int nb_rotated_units{ int(f * readout.get_nb_channels()) };
    const double angle_in_rad{ angle_in_degrees * M_PI / 180. };
    
    Matrix rotation_matrix(_dim, _dim);
    rotation_matrix << cos(angle_in_rad), sin(-angle_in_rad), sin(angle_in_rad), cos(angle_in_rad);
    
    for (int r=0; r<nb_rotated_units; r++){
        P.row(r) = P.row(r) * rotation_matrix.transpose();
    }
}
