/**
 * RNN.cpp | bmi_model
 *
 * TODO: Furious need for a clean-up + better commenting.
 */
#include "RNN.hpp"
#include "rand_mat.hpp"
#include <assert.h>
#include <iomanip>

RNN::RNN(int Nin, int Nrec, int Nout, double noise, double tauM, double alpha_reward, ActivationType phi, double output_decimation) : _Nin(Nin), _Nrec(Nrec), _Nout(Nout), _tau(tauM), _noise(noise), _alpha_reward(alpha_reward), phi(phi) {
        
    _a = dt/tauM;
    _scale_initial_potential = 1.;
    
    if (phi == ActivationType::relu) {
        U = random_uniform_matrix(_Nrec, _Nin, sqrt(1./_Nin));
        W = random_gaussian_matrix(_Nrec, _Nrec, sqrt(1./_Nrec)); //Matrix::Zero(_Nrec, _Nrec);//
        b = random_uniform_vector(_Nrec, 0.5, 1.5); //Vector::Zero(_Nrec);//random_uniform_vector(_Nrec, 0.5, 1.5);//Vector::Ones(_Nrec);
        V = random_gaussian_matrix(_Nout, _Nrec, 0.2/_Nrec);  // DEBUG!!! was 0.1/_Nrec
        //V = balanced_uniform_matrix(_Nout, _Nrec, 0.5/_Nrec);
        hold_signal = -1*Vector::Ones(_Nrec); //random_uniform_vector(_Nrec, -0.5, 0); //-0.5*Vector::Ones(_Nrec);
    }
    else if (phi == ActivationType::sigmoid){
        U = random_uniform_matrix(_Nrec, _Nin, 1./sqrt(_Nin));
        W = 2*random_gaussian_matrix(_Nrec, _Nrec, 1./sqrt(_Nrec));
        b = -0.5*W * Vector::Ones(_Nrec); //Vector::Zero(_Nrec); //-0.5 * W * Vector::Ones(_Nrec);//Vector::Zero(_Nrec); ////
        V = 2*balanced_gaussian_matrix(_Nout, _Nrec, sqrt(M_PI_2)/_Nrec);
        //V = random_gaussian_matrix(_Nout, _Nrec, 2*sqrt(M_PI_2)/_Nrec);
        hold_signal = 0.5 * Vector::Ones(_Nrec);
    }
    else if (phi == ActivationType::tanh || phi == ActivationType::linear){
        U = random_uniform_matrix(_Nrec, _Nin, 1./sqrt(_Nin));
        W = random_gaussian_matrix(_Nrec, _Nrec, 1./sqrt(_Nrec));
        b = Vector::Zero(_Nrec);
        V = random_gaussian_matrix(_Nout, _Nrec, sqrt(M_PI_2)/_Nrec);
        //V = balanced_gaussian_matrix(_Nout, _Nrec, sqrt(M_PI_2)/_Nrec);
        hold_signal = 0.5 * Vector::Ones(_Nrec);
    }
    else if (phi == ActivationType::retanh) {
        U = random_uniform_matrix(_Nrec, _Nin, 1./sqrt(_Nin));
        W = random_gaussian_matrix(_Nrec, _Nrec, 1./sqrt(_Nrec));
        b = 0.5*Vector::Ones(_Nrec); //Vector::Zero(_Nrec); // random_uniform_vector(_Nrec, 0., 0.5);
        V = random_gaussian_matrix(_Nout, _Nrec, 3./_Nrec);
        hold_signal = random_uniform_vector(_Nrec, -0.5, 0.5);
    }
    m_U = Matrix::Zero(_Nrec, _Nin);
    v_U = Matrix::Zero(_Nrec, _Nin);

    m_W = Matrix::Zero(_Nrec, _Nrec);
    v_W = Matrix::Zero(_Nrec, _Nrec);
    
    m_b = Vector::Zero(_Nrec);
    v_b = Vector::Zero(_Nrec);
    
    v_init = Vector::Zero(_Nrec);  // DEBUG!!!!
    //v_init = find_initial_membrane_potential();
    
    if (output_decimation > 0.){
        double norm_V_full = V.norm();
        std::vector<int> all_units(_Nrec);
        std::iota(all_units.begin(), all_units.end(), 0);
        std::shuffle(all_units.begin(), all_units.end(), myrng);
        
        const int n_units_to_read_out{ int(output_decimation * _Nrec) };
        for (int i{ 0 }; i<n_units_to_read_out; ++i) V.col(all_units[i]) = Vector::Zero(_Nout);
        V = norm_V_full * V / V.norm();
    }
}

double RNN::loss_for_v_init(const Vector &v){
    Vector u = V * activate(v, phi);
    return 0.5 * u.squaredNorm();
}

Vector RNN::find_initial_membrane_potential(){
    double lr{ 1. };
    Vector v = Vector::Zero(_Nrec);
    double r_min{0.};
    double r_max{1.};
    Matrix Phi = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(_Nrec);

    std::cout << "\nFinding initial condition..." << std::endl;
    
    switch (phi) {
        case ActivationType::sigmoid:
            r_min = 0.3; r_max = 0.6;
            v = random_uniform_vector(_Nrec, inverse(r_min, phi), inverse(r_max, phi));
            break;
        case ActivationType::relu:
            r_min = 0.5; r_max = 1.5;
            v = random_uniform_vector(_Nrec, r_min, r_max);
            break;
        case ActivationType::retanh:
            r_min = 0.1; r_max = 0.5;
            v = random_uniform_vector(_Nrec, r_min, r_max);
            break;
        case ActivationType::linear:
            r_min = -0.5; r_max = 0.5;
            v = random_uniform_vector(_Nrec, r_min, r_max);
            break;
        case ActivationType::tanh:
            r_min = -0.5; r_max = 0.5;
            v = random_uniform_vector(_Nrec, r_min, r_max);
            break;
    }
    
    double inverse_r_max = inverse(r_max, phi);
    double inverse_r_min = inverse(r_min, phi);
    Vector r = activate(v, phi);
    int i{ 0 };
    while ((V * r).norm() > 1e-10 ){
        //if (i % 10000) {
        //    std::cout << "in loop.." << std::endl;
        //    std::cout << "Norm = " << (V * r).norm() << std::endl;
        //}
        Phi.diagonal() = derivate(activate(v, phi), phi);
        v = v - lr * Phi * V.transpose() * V * activate(v, phi);
        r = activate(v, phi);
        for (int elem=0; elem<r.size(); elem++) {
            v(elem) = r(elem) > r_max ? inverse_r_max : v(elem);
            v(elem) = r(elem) < r_min ? inverse_r_min : v(elem);
        }
        i++;
    }
    
    if ((V * r).norm() < 1e-10) std::cout << "... initial condition found.\n";
    else {
        std::cout << "Initial condition NOT found...\n";
        std::cout << "Norm of controls = " << (V * r).norm() << '\n';
    }
    //std::cout << v.transpose() << std::endl << std::endl;
    return v;
}

Vector RNN::set_initial_potential(const double scale){
    Vector v0(_Nrec);
    return scale * v0.setRandom();
}

void RNN::transform_to_sigmoid(){
    if (phi == ActivationType::tanh){
        W *= 2.;
        b -= W * Vector::Ones(_Nrec);
        phi = ActivationType::sigmoid;
    } else {
        std::cerr << "Activation function is not tanh. Nothing to transform.\n";
    }
}

void RNN::transform_to_tanh(){
    if (phi == ActivationType::sigmoid){
        W /= 2.;
        b += 0.5 * W * Vector::Ones(_Nrec);
        phi = ActivationType::tanh;
    } else {
        std::cerr << "Activation function is not sigmoid. Nothing to transform.\n";
    }
}


Matrix RNN::forward(Input& x, int target_id) {
    const int duration = x.get_duration();
    const int hold_duration = x.get_hold_duration();
    Matrix controls(_Nout, duration+hold_duration);
    Vector v(_Nrec), r(_Nrec), u(_Nout);
    
    const int nb_targets{ x.get_nb_targets()};
    Vector target_direction = get_target_direction_from_id(target_id, nb_targets, x.imp->get_dim());

    x.reset();
    
    v = set_initial_potential(_scale_initial_potential);
    r = activate(v, phi);
    u = V * r;
        
    for (int t(-hold_duration); t<duration; t++) {
        Vector u_prev = u;
        Vector xi = (_noise / sqrt(_a)) * random_gaussian_vector(_Nrec);
        Vector hold_input = static_cast<double>(t < 0) * hold_signal;
        //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;
        v = (1 - _a) * v + _a * (U * x(target_id) + W * r + b + xi + hold_input);
        r = activate(v, phi);
        u = V * r;
        x.next(u_prev, r, target_direction);
        controls.col(t+hold_duration) = u_prev;
    }
    return controls;
}

Matrix RNN::forward(Input& x, int target_id, Monitor & m) {
    const int duration = x.get_duration();
    const int hold_duration = x.get_hold_duration();
    Matrix controls(_Nout, duration+hold_duration);
    Vector v(_Nrec), r(_Nrec), u(_Nout);
    
    const int nb_targets{ x.get_nb_targets()};
    Vector target_direction = get_target_direction_from_id(target_id, nb_targets, x.imp->get_dim());
    
    x.reset();
    v = set_initial_potential(_scale_initial_potential);
    r = activate(v, phi);
    u = V * r;
        
    for (int t(-hold_duration); t<duration; t++) {
        Vector u_prev = u;
        Vector xi = (_noise / sqrt(_a)) * random_gaussian_vector(_Nrec);
        Vector hold_input = static_cast<double>(t < 0) * hold_signal;
        //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;
        v = (1 - _a) * v + _a * (U * x(target_id) + W * r + b + xi + hold_input);
        r = activate(v, phi);
        u = V * r;
        
        m.record_snapshot(r);
        controls.col(t+hold_duration) = u_prev;

        x.next(u_prev, r, target_direction);
    }

    return controls;
}

Matrix RNN::forward(Input& x, int target_id, Monitor & m_rec, Monitor & m_input) {
    const int duration = x.get_duration();
    const int hold_duration = x.get_hold_duration();
    Matrix controls(_Nout, duration+hold_duration);
    Vector v(_Nrec), r(_Nrec), u(_Nout);
    
    const int nb_targets{ x.get_nb_targets()};
    Vector target_direction = get_target_direction_from_id(target_id, nb_targets, x.imp->get_dim());
    
    x.reset();
    v = set_initial_potential(_scale_initial_potential);
    r = activate(v, phi);
    u = V * r;
    
    for (int t(-hold_duration); t<duration; t++) {
        Vector u_prev = u;
        Vector xi = (_noise / sqrt(_a)) * random_gaussian_vector(_Nrec);
        //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;
        Vector hold_input = static_cast<double>(t < 0) * hold_signal;
        v = (1 - _a) * v + _a * (U * x(target_id) + W * r + b + xi + hold_input);
        r = activate(v, phi);
        u = V * r;

        m_rec.record_snapshot(r);
        m_input.record_snapshot(x(target_id));
        controls.col(t+hold_duration) = u_prev;

        x.next(u_prev, r, target_direction);
    }
    
    return controls;
}

Matrix RNN::forward(Input& x, int target_id, Monitor & m_rec, Monitor & m_input, Monitor & m_state) {
    const int duration = x.get_duration();
    const int hold_duration = x.get_hold_duration();
    Matrix controls(_Nout, duration+hold_duration);
    Vector v(_Nrec), r(_Nrec), u(_Nout);
    
    const int nb_targets{ x.get_nb_targets()};
    Vector target_direction = get_target_direction_from_id(target_id, nb_targets, x.imp->get_dim());
    x.reset();
    v = set_initial_potential(_scale_initial_potential);
    r = activate(v, phi);
    u = V * r;
    
    for (int t(-hold_duration); t<duration; t++) {
        Vector u_prev = u;
        Vector xi = (_noise / sqrt(_a)) * random_gaussian_vector(_Nrec);
        //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;
        Vector hold_input = static_cast<double>(t < 0) * hold_signal;
        v = (1 - _a) * v + _a * (U * x(target_id) + W * r + b + xi + hold_input);
        r = activate(v, phi);
        u = V * r;

        m_rec.record_snapshot(r);
        m_input.record_snapshot(x(target_id));
        m_state.record_snapshot(x.imp->state);
        controls.col(t+hold_duration) = u_prev;

        x.next(u_prev, r, target_direction);
    }
    
    return controls;
}

double RNN::compute_mean_reward(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int nb_reals){
    int nb_targets(in.get_nb_targets());
    double reward_traces_placeholder[nb_targets];
    double reward_traces_prep_placeholder[nb_targets];
    double mean_reward{ 0. };
    for (int i=0; i<nb_reals; i++) mean_reward += _compute_reward_only(loss, imp, in, target, p, reward_traces_placeholder, reward_traces_prep_placeholder);
    return mean_reward/nb_reals;
}

double RNN::compute_median_reward(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int nb_reals){
    int nb_targets(in.get_nb_targets());
    double reward_traces_placeholder[nb_targets];
    double reward_traces_prep_placeholder[nb_targets];
    double median_reward{ 0. };
    std::vector<double> rewards(nb_reals);
    for (int i=0; i<nb_reals; i++) {
        double r = _compute_reward_only(loss, imp, in, target, p, reward_traces_placeholder, reward_traces_prep_placeholder);
        rewards[i] = r;
    }
    std::sort(rewards.begin(), rewards.end());
    if (nb_reals % 2 == 0) median_reward = 0.5 * (rewards[nb_reals/2 - 1] + rewards[nb_reals/2]);
    else median_reward = rewards[(nb_reals-1)/2];
    return median_reward;
}


void RNN::_weight_update(double lr[4], const GradientRecurrent& g, int epoch, OptimizerType optimizer){
    if (optimizer == OptimizerType::Adam) {
        constexpr double beta1 { 0.9 };
        constexpr double beta2 { 0.999 };
        update_moments(g, beta1, beta2);
        double beta1t{pow(beta1, epoch+1)};
        double beta2t{pow(beta2, epoch+1)};
        U = U.array() + lr[0]*sqrt(1.-beta2t)/(1.-beta1t) * m_U.array()/(Eigen::sqrt(v_U.array()) + 1e-10);
        W = W.array() + lr[1]*sqrt(1.-beta2t)/(1.-beta1t) * m_W.array()/(Eigen::sqrt(v_W.array()) + 1e-10);
        b = b.array() + lr[3]*sqrt(1.-beta2t)/(1.-beta1t) * m_b.array()/(Eigen::sqrt(v_b.array()) + 1e-10);
        //if (epoch > 1000 and epoch % 1000 == 0) std::cout << sqrt(1.-beta2t)/(1.-beta1t) * m_W.array()/(Eigen::sqrt(v_W.array()) + 1e-10) << std::endl;
    } else if (optimizer == OptimizerType::SGD) {
        if (lr[0] > 0) U += lr[0] * g.U;
        W += lr[1] * g.W;
        b += lr[3] * g.b;
    }
}

void RNN::update_moments(const GradientRecurrent& g, double beta1, double beta2){
    m_U = beta1 * m_U + (1 - beta1) * g.U;
    m_W = beta1 * m_W + (1 - beta1) * g.W;
    m_b = beta1 * m_b + (1 - beta1) * g.b;
    
    v_U = beta2 * v_U.array() + (1 - beta2) * g.U.array() * g.U.array();
    v_W = beta2 * v_W.array() + (1 - beta2) * g.W.array() * g.W.array();
    v_b = beta2 * v_b.array() + (1 - beta2) * g.b.array() * g.b.array();
}

void RNN::reset_moments(){
    m_U.setZero();
    m_W.setZero();
    m_b.setZero();
    v_U.setZero();
    v_W.setZero();
    v_b.setZero();
}

std::pair<GradientRecurrent, double> RNN::_compute_gradient(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, int target_id, double *reward_trace, double *reward_traces_prep){
    
    double effort_cost{ 0. };
    double activity_cost{ 0. };
    double objective_cost{ 0. };
    const int duration(in.get_duration());
    const int hold_duration(in.get_hold_duration());
    
    Vector v(_Nrec), r(_Nrec), u(_Nout);                      // network variables
    Vector center_target = Vector::Zero(imp->get_dim());      // target during hold (motionless effector at center)
    
    const int nb_targets{ in.get_nb_targets()};
    Vector target_direction = get_target_direction_from_id(target_id, nb_targets, in.imp->get_dim());
    
    // For dynamic input, check whether pointer to Effector in Input is the same address as imp
    if (in.get_input_type() == DYNAMIC || in.get_input_type()==FFNINPUT) {
        if (imp != in.imp) std::runtime_error("Effector in dynamic input does not point to effector used by network.");
    }
    
    // Initialize effector and input
    imp->reset();
    in.reset();
    
    // Initialize network variables
    v = set_initial_potential(_scale_initial_potential);
    r = activate(v, phi);
    u = V * r;
    
    // Gradient and eligibility traces
    GradientRecurrent grad(_Nin, _Nrec, _Nout);
    EligibilityTraceRecurrent et(_Nin, _Nrec, _Nout);

    // Run hold-to-center interval
    for (int t(-hold_duration); t<0; t++) {
        // Forward
        Vector r_prev = r;
        Vector u_prev = u;
        Vector xi = random_gaussian_vector(_Nrec);
        Vector hold_input = hold_signal;
        //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;

        v = (1 - _a) * v + _a * (U * in(target_id) + W * r_prev + b + hold_input) + _noise*sqrt(_a)*xi;  // note that hold signal is only present here
        r = activate(v, phi);
        u = V * r;
        
        // Running effort cost on arm motion and activity
        effort_cost += 0.5 * loss->effort_penalty * u_prev.squaredNorm();
        activity_cost += loss->activity_regularizer * r_prev.squaredNorm()/double(_Nrec);
        
        // Update traces
        if (t < -1) {
            if(p.learning_rate[0] > 0) et.U += xi * in(target_id).transpose();
            et.W += xi * r_prev.transpose();
            et.b += xi;
        }
        // Update input
        in.next(u_prev, r, target_direction);
        
        // Objective cost
        objective_cost += (*loss)(in.imp->end_effector, center_target, -1-t, p.scale_factor_hold);
    }
    double reward_prep{ 0. };
    if (hold_duration > 0){
        // Reward for hold-to-center interval
        reward_prep = -(activity_cost+effort_cost)/hold_duration - objective_cost;
        // Gradient update
        grad.update(reward_prep, *reward_traces_prep, et);
        // Update reward trace for hold-to-center interval
        *reward_traces_prep = _alpha_reward * *reward_traces_prep + (1 - _alpha_reward) * reward_prep;
        if (_long_reward_traces_preparatory.size() > 0) _long_reward_traces_preparatory[target_id] = *reward_traces_prep;
    }
    
    // Run post-go-cue interval
    activity_cost = 0.;
    effort_cost = 0.;
    objective_cost = 0.;
    et.reset();
    for (int t(0);t<duration;t++) {
        // Forward
        Vector r_prev = r;
        Vector u_prev = u;
        Vector xi = random_gaussian_vector(_Nrec);
        v = (1 - _a) * v + _a * (U * in(target_id) + W * r_prev + b) + _noise*sqrt(_a)*xi;
        r = activate(v, phi);
        u = V * r;
        
        // Running loss
        effort_cost += 0.5 * loss->effort_penalty * u_prev.squaredNorm();
        activity_cost += loss->activity_regularizer * r_prev.squaredNorm()/double(_Nrec);
        
        // Update traces
        if (t < duration - 1) {
            if(p.learning_rate[0] > 0) et.U += xi * in(target_id).transpose();
            et.W += xi * r_prev.transpose();
            et.b += xi;
        }
        
        // Update input
        in.next(u_prev, r, target_direction);
        
        // Objective cost
        objective_cost += (*loss)(in.imp->end_effector, target(target_id), duration-1-t);
    }
    double reward{ 0. };
    if (duration > 0){
        // Reward
        reward = -(activity_cost+effort_cost)/duration - objective_cost;
        grad.update(reward, *reward_trace, et);
        //std::cout << "Activity reg = " << activity_cost/duration << std::endl;
        //std::cout << "Effort cost = " << effort_cost/duration << std::endl;
        //std::cout << "Objective cost = " << objective_cost << std::endl;
        // Update reward trace
        *reward_trace = _alpha_reward * *reward_trace + (1 - _alpha_reward) * reward;
        if (_long_reward_traces.size() > 0) _long_reward_traces[target_id] = *reward_trace;
    }
    auto grad_and_reward = std::make_pair(grad, reward_prep+reward);
    return grad_and_reward;
}

double RNN::_compute_reward_only(Loss * loss, Effector * imp, Input & in, Target & target, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[]){
    double total_reward{ 0. }; // sum of rewards across examples
    const int duration(in.get_duration());
    const int hold_duration(in.get_hold_duration());
    const int nb_targets(target.get_nb_targets());

    Vector v(_Nrec), r(_Nrec), u(_Nout);                      // network variables
    Vector center_target = Vector::Zero(imp->get_dim());      // target during hold (motionless effector at center)
    
    
    // For dynamic input, check whether in pointer to Effector in Input is the same address as imp
    if (in.get_input_type() == DYNAMIC || in.get_input_type()==FFNINPUT) {
        if (imp != in.imp) std::runtime_error("Effector in dynamic input does not point to implement used by network.");
    }
    
    for (int i(0);i<nb_targets; i++){
        double effort_cost = 0.;
        double objective_cost = 0.;
        double activity_cost = 0.;
        
        // Initialize effector and input
        imp->reset();
        in.reset();
        
        // Initialize network variables
        v = set_initial_potential(_scale_initial_potential);
        r = activate(v, phi);
        u = V * r;
        
        //Target direction, if effector is PassiveCursor
        //std::cout << "Dim for imp = " << imp->get_dim() << '\n';
        //std::cout << "Dim for in = " << in.imp->get_dim() << '\n';
        Vector target_direction = get_target_direction_from_id(i, nb_targets, in.imp->get_dim());

        // Run hold-to-center interval
        for (int t(-hold_duration); t<0; t++) {
            // Forward
            Vector r_prev = r;
            Vector u_prev = u;
            Vector xi = random_gaussian_vector(_Nrec);
            Vector hold_input = hold_signal;
            //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;
            
            v = (1 - _a) * v + _a * (U * in(i) + W * r_prev + b + hold_input) + _noise*sqrt(_a)*xi;  // note that hold signal is only present here
            r = activate(v, phi);
            u = V * r;

            // Running effort cost on arm motion
            effort_cost += 0.5 * loss->effort_penalty * u_prev.squaredNorm();
            activity_cost += loss->activity_regularizer * r_prev.squaredNorm()/double(_Nrec);
            
            // Update input
            in.next(u_prev, r, target_direction);
            
            // Objective cost
            objective_cost += (*loss)(in.imp->end_effector, center_target, -1-t, p.scale_factor_hold);
        }
        
        if (hold_duration > 0){
            // Reward for hold-to-center interval
            double reward_prep = -(activity_cost+effort_cost)/hold_duration - objective_cost;
            //std::cout << "reward prep = " << reward_prep << std::endl;
            reward_traces_prep[i] = _alpha_reward * reward_traces_prep[i] + (1 - _alpha_reward) * reward_prep;
            total_reward += reward_prep;
            if (_long_reward_traces_preparatory.size() > 0) _long_reward_traces_preparatory[i] = reward_traces_prep[i];
        }
        // Run post-go-cue interval
        activity_cost = 0.;
        effort_cost = 0.;
        objective_cost = 0.;
        for (int t(0); t<duration; t++) {
            // Forward
            Vector r_prev = r;
            Vector u_prev = u;
            Vector xi = random_gaussian_vector(_Nrec);
            Vector hold_input = Vector::Zero(_Nrec);
            //Vector hold_input = (smoothstep(-hold_duration, -hold_duration + hold_duration/10, t) - smoothstep(0, hold_duration/10, t))*hold_signal;

            v = (1 - _a) * v + _a * (U * in(i) + W * r_prev + b + hold_input) + _noise*sqrt(_a)*xi;
            r = activate(v, phi);
            u = V * r;

            // Running loss
            effort_cost += 0.5 * loss->effort_penalty * u_prev.squaredNorm();
            activity_cost += loss->activity_regularizer * r_prev.squaredNorm()/double(_Nrec);

            // Update input
            in.next(u_prev, r, target_direction);

            // Objective cost
            objective_cost += (*loss)(in.imp->end_effector, target(i), duration-1-t);
        }
        if (duration > 0){
            // Reward
            double reward = -(activity_cost+effort_cost)/duration - objective_cost;
            //std::cout << "Activity reg = " << activity_cost/duration << std::endl;
            //std::cout << "Effort cost = " << effort_cost/duration << std::endl;
            //std::cout << "Objective cost = " << objective_cost << std::endl;

            //std::cout << "reward = " << reward << std::endl;
            reward_traces[i] = _alpha_reward * reward_traces[i] + (1 - _alpha_reward) * reward;
            total_reward += reward;
            if (_long_reward_traces.size() > 0) _long_reward_traces[i] = reward_traces[i];
        }
    }
    //std::cout << "Mean reward in compute_reward_only = " << total_reward / nb_targets << std::endl;
    return total_reward / nb_targets;
}

double RNN::_train_epoch(Loss * loss, Effector * imp,  Input & inputs, Target & targets, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[], int epoch, OptimizerType optimizer) {
    int nb_targets(inputs.get_nb_targets());
    double total_reward(0);
    
    if (!p.learning_after_each_example) {
        GradientRecurrent total_grad(_Nin, _Nrec, _Nout);
        for (int i(0);i<nb_targets; i++){
            auto grad_and_reward = _compute_gradient(loss, imp, inputs, targets, p, i, reward_traces+i, reward_traces_prep+i); //updates gradient and reward
            total_grad += std::get<GradientRecurrent>(grad_and_reward);
            total_reward += std::get<double>(grad_and_reward);
        }
        _weight_update(p.learning_rate, total_grad, epoch, optimizer);
    }
    else {
        for (int i(0);i<nb_targets; i++){
            GradientRecurrent grad(_Nin, _Nrec, _Nout);
            auto grad_and_reward = _compute_gradient(loss, imp, inputs, targets, p, i, reward_traces+i, reward_traces_prep+i);
            grad = std::get<GradientRecurrent>(grad_and_reward);
            _weight_update(p.learning_rate, grad, epoch, optimizer);
            total_reward += std::get<double>(grad_and_reward);
        }
    }
    return total_reward/nb_targets;
}

double RNN::_train_epoch_with_dropout(Loss * loss, VelocityKalmanFilter &kf,  Input & inputs, Target & targets, TrainParamsRNN & p, double reward_traces[], double reward_traces_prep[], int epoch, OptimizerType optimizer, const double dropout_prob) {
    int nb_targets(inputs.get_nb_targets());
    double total_reward(0);
    std::uniform_int_distribution<size_t> UID(0, kf.get_nb_channels()-1); // for dropout
    std::uniform_real_distribution<double> unif_real(0., 1.);
        
    if (!p.learning_after_each_example) {
        GradientRecurrent total_grad(_Nin, _Nrec, _Nout);
        VelocityKalmanFilter kf_copy{kf};
        std::vector<size_t> readout_unit_ids = kf_copy.get_readout_unit_ids();
        for (auto readout_id : readout_unit_ids){
            const double r{ unif_real(rng_for_dropout) };
            if (r < dropout_prob) kf_copy.remove_unit(readout_id);
        }
        //size_t unit_to_drop = UID(rng_for_dropout);
        //kf_copy.remove_unit(unit_to_drop);
        inputs.set_context(&kf_copy);
        
        for (int i(0);i<nb_targets; i++){
            auto grad_and_reward = _compute_gradient(loss, &kf_copy, inputs, targets, p, i, reward_traces+i, reward_traces_prep+i); //updates gradient and reward
            total_grad += std::get<GradientRecurrent>(grad_and_reward);
            total_reward += std::get<double>(grad_and_reward);
        }
        _weight_update(p.learning_rate, total_grad, epoch, optimizer);
    }
    else {
        for (int i(0);i<nb_targets; i++){
            GradientRecurrent grad(_Nin, _Nrec, _Nout);
            VelocityKalmanFilter kf_copy{kf};
            size_t unit_to_drop = UID(rng_for_dropout);
            kf_copy.remove_unit(unit_to_drop);
            inputs.set_context(&kf_copy);
            auto grad_and_reward = _compute_gradient(loss, &kf_copy, inputs, targets, p, i, reward_traces+i, reward_traces_prep+i);
            grad = std::get<GradientRecurrent>(grad_and_reward);
            _weight_update(p.learning_rate, grad, epoch, optimizer);
            total_reward += std::get<double>(grad_and_reward);
        }
    }
    inputs.set_context(&kf); // reset correct context
    return total_reward/nb_targets;
}



void RNN::train(Loss * loss, Effector * imp, Input & in, Target & t, TrainParamsRNN & p,
                std::string file_prefix, std::ofstream::openmode mode, OptimizerType optimizer){
    
    if (in.get_nb_targets() != t.get_nb_targets()){
        throw std::runtime_error("Unequal number of targets for input and target.");
    }
    int nb_targets = in.get_nb_targets();
    
    double single_epoch_loss{ 0. };
    double initial_learning_rate{p.learning_rate[1]};
    double asymptotic_learning_rate{ initial_learning_rate/5. };
    
    // Preparing output file for the losses
    std::ofstream outfile;
    if (file_prefix.size() > 0) outfile.open(file_prefix+"loss.txt", mode);
        
    double reward_traces[nb_targets];
    double reward_traces_prep[nb_targets];
    
    if (_reward_was_memorized()){
        for (int i(0);i<nb_targets;i++) reward_traces[i] = _long_reward_traces[i];
        for (int i(0);i<nb_targets;i++) reward_traces_prep[i] = _long_reward_traces_preparatory[i];
    } else {// Reset reward traces if not memorized.
        for (int i(0);i<nb_targets;i++) reward_traces[i] = 0;
        for (int i(0);i<nb_targets;i++) reward_traces_prep[i] = 0;
        // Reset moments of the gradient
        reset_moments();
    }
    
    if (_reward_was_memorized()) {
        for (int i(0);i<p.nb_epochs;i++){
            if (p.lr_adapt) {
                if (p.learning_rate[0] > asymptotic_learning_rate){
                    for (int lr_i=0; lr_i<4; lr_i++) p.learning_rate[lr_i] *= p.lr_adaptation_factor;
                }
            }
            single_epoch_loss = _train_epoch(loss, imp, in, t, p, reward_traces, reward_traces_prep, i, optimizer);
            
            if (p.verbose){
                if (i % p.interval_verbose == 0){
                    std::cout << "Epoch " << i+1 << " loss = " << -single_epoch_loss << '\n';
                }
            }
            if (file_prefix.size() > 0) outfile << i << "\t" << -single_epoch_loss << '\n';
        }
    } else {
        for (int i(0);i<p.nb_epochs;i++){
            if (p.lr_adapt) {
                if (p.learning_rate[0] > asymptotic_learning_rate){
                    for (int lr_i=0; lr_i<4; lr_i++) p.learning_rate[lr_i] *= p.lr_adaptation_factor;
                }
            }
            if (i > 5) single_epoch_loss = _train_epoch(loss, imp, in, t, p, reward_traces, reward_traces_prep, i, optimizer);
            else single_epoch_loss = _compute_reward_only(loss, imp, in, t, p, reward_traces, reward_traces_prep);
            
            if (p.verbose){
                if (i % p.interval_verbose == 0){
                    std::cout << "Epoch " << i+1 << " loss = " << -single_epoch_loss << " | learning rate = " << p.learning_rate[0] << '\n';
                }
            }
            
            if (file_prefix.size() > 0) outfile << i << "\t" << -single_epoch_loss << '\n';
        }
    }
}

void RNN::train_with_dropout(Loss * loss, VelocityKalmanFilter &kf, Input & in, Target & t, TrainParamsRNN & p,
                std::string file_prefix, std::ofstream::openmode mode, OptimizerType optimizer){
    
    if (in.get_nb_targets() != t.get_nb_targets()){
        throw std::runtime_error("Unequal number of targets for input and target.");
    }
    int nb_targets = in.get_nb_targets();
    
    double single_epoch_loss{ 0. };
    double initial_learning_rate{p.learning_rate[1]};
    double asymptotic_learning_rate{ initial_learning_rate/5. };
    
    // Preparing output file for the losses
    std::ofstream outfile;
    if (file_prefix.size() > 0) outfile.open(file_prefix+"loss.txt", mode);
        
    double reward_traces[nb_targets];
    double reward_traces_prep[nb_targets];
    
    if (_reward_was_memorized()){
        for (int i(0);i<nb_targets;i++) reward_traces[i] = _long_reward_traces[i];
        for (int i(0);i<nb_targets;i++) reward_traces_prep[i] = _long_reward_traces_preparatory[i];
    } else {// Reset reward traces if not memorized.
        for (int i(0);i<nb_targets;i++) reward_traces[i] = 0;
        for (int i(0);i<nb_targets;i++) reward_traces_prep[i] = 0;
        // Reset moments of the gradient
        reset_moments();
    }
    
    if (_reward_was_memorized()) {
        for (int i(0);i<p.nb_epochs;i++){
            if (p.lr_adapt) {
                if (p.learning_rate[0] > asymptotic_learning_rate){
                    for (int lr_i=0; lr_i<4; lr_i++) p.learning_rate[lr_i] *= p.lr_adaptation_factor;
                }
            }
            single_epoch_loss = _train_epoch_with_dropout(loss, kf, in, t, p, reward_traces, reward_traces_prep, i, optimizer);
            
            if (p.verbose){
                if (i % p.interval_verbose == 0){
                    std::cout << "Epoch " << i+1 << " loss = " << -single_epoch_loss << '\n';
                }
            }
            if (file_prefix.size() > 0) outfile << i << "\t" << -single_epoch_loss << '\n';
        }
    } else {
        for (int i(0);i<p.nb_epochs;i++){
            if (p.lr_adapt) {
                if (p.learning_rate[0] > asymptotic_learning_rate){
                    for (int lr_i=0; lr_i<4; lr_i++) p.learning_rate[lr_i] *= p.lr_adaptation_factor;
                }
            }
            if (i > 5) single_epoch_loss = _train_epoch_with_dropout(loss, kf, in, t, p, reward_traces, reward_traces_prep, i, optimizer);
            else single_epoch_loss = _compute_reward_only(loss, &kf, in, t, p, reward_traces, reward_traces_prep);
            
            if (p.verbose){
                if (i % p.interval_verbose == 0){
                    std::cout << "Epoch " << i+1 << " loss = " << -single_epoch_loss << " | learning rate = " << p.learning_rate[0] << '\n';
                }
            }
            
            if (file_prefix.size() > 0) outfile << i << "\t" << -single_epoch_loss << '\n';
        }
    }
}


void RNN::init_long_reward_traces(int nb_targets){
    for (int i(0);i<nb_targets;i++) _long_reward_traces.push_back(0.);
    for (int i(0);i<nb_targets;i++) _long_reward_traces_preparatory.push_back(0.);
}

void RNN::reset_long_reward_traces(){
    _long_reward_traces.clear();
    _long_reward_traces_preparatory.clear();
}

bool RNN::_reward_was_memorized(){
    bool has_memory = false;
    if (!_long_reward_traces.empty()){
        for (int i(0);i<_long_reward_traces.size();i++){
            if (abs(_long_reward_traces[i]) > 1e-12) has_memory = true;
        }
    }
    return has_memory;
}

void RNN::save(std::string file_prefix){
    std::ofstream f_U(file_prefix + "_U.txt");
    std::ofstream f_W(file_prefix + "_W.txt");
    std::ofstream f_V(file_prefix + "_V.txt");
    std::ofstream f_b(file_prefix + "_b.txt");
    
    for (int r(0);r<U.rows();r++) {
        for (int c(0);c<U.cols()-1;c++) {
            f_U <<U(r, c)<< ",";
        }
        f_U <<U(r, U.cols()-1)<< std::endl;
    }
    
    for (int r(0);r<W.rows();r++) {
        for (int c(0);c<W.cols()-1;c++) {
            f_W <<W(r, c)<< ",";
        }
        f_W <<W(r, W.cols()-1)<< std::endl;
    }
    
    for (int r(0);r<V.rows();r++) {
        for (int c(0);c<V.cols()-1;c++) {
            f_V <<V(r, c)<< ",";
        }
        f_V <<V(r, V.cols()-1)<< std::endl;
    }
    
    for (int r(0);r<b.size()-1;r++){
        f_b << b.data()[r] << ",";
    }
    f_b << b(b.size()-1) << std::endl;

    
    f_U.close();
    f_W.close();
    f_V.close();
    f_b.close();
    
}

void RNN::load(std::string file_prefix){
    std::ifstream file;
    std::string line;
    double val;
    int row_counter, col_counter;
    
    //Checkout network size
    int network_size=0;
    file.open(file_prefix + "_W.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file: " + file_prefix + "_W.txt");
    while (std::getline(file, line)){
        network_size++;
    }
    file.close();
    
    //Check number of inputs
    int input_size=0;
    file.open(file_prefix + "_U.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    std::getline(file, line);
    std::stringstream ss(line);
    while (ss >> val){
        input_size++;
        if(ss.peek() == ',') ss.ignore();
    }
    file.close();
    
    //Check output size
    int output_size=0;
    file.open(file_prefix + "_V.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    while (std::getline(file, line)){
        output_size++;
    }
    file.close();
    
    U = Matrix(network_size, input_size);
    W = Matrix(network_size, network_size);
    V = Matrix(output_size, network_size);
    b = Vector(network_size);
    
    //Loading U
    file.open(file_prefix + "_U.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter = 0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            U(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading W
    file.open(file_prefix + "_W.txt");
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
    
    //Loading V
    file.open(file_prefix + "_V.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            V(row_counter, col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
    
    //Loading b
    file.open(file_prefix + "_b.txt");
    if(!file.is_open()) throw std::runtime_error("Could not open file");
    row_counter=0;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        col_counter = 0;
        while (ss >> val){
            b(col_counter) = val;
            if(ss.peek() == ',') ss.ignore();
            col_counter++;
        }
        row_counter++;
    }
    file.close();
}
