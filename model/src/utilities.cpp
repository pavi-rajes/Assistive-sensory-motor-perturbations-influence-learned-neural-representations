/**
 * utilities.cpp | bmi_model
 */
#include "utilities.hpp"
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include "Target.hpp"

Eigen::Vector2d condition_end_effector_in_joint_angles(const Eigen::Vector2d& angle, const Eigen::Vector2d& end_effector_position, const double l1, const double l2){
    Eigen::Vector2d f;
    f(0) = l1 * cos(angle(0)) + l2 * cos(angle(0) + angle(1)) - end_effector_position(0);
    f(1) = l1 * sin(angle(0)) + l2 * sin(angle(0) + angle(1)) - end_effector_position(1);
    return f;
}

Eigen::Matrix2d jacobian(const Eigen::Vector2d& angle, const double l1, const double l2){
    Eigen::Matrix2d J;
    J(0,0) = -l1 * sin(angle(0)) - l2 * sin(angle(0) + angle(1));
    J(0,1) = -l2 * sin(angle(0) + angle(1));
    J(1,0) = l1 * cos(angle(0)) + l2 * cos(angle(0) + angle(1));
    J(1,1) = l2 * cos(angle(0) + angle(1));
    return J;
}

Eigen::Matrix2d djacobiandt(const Eigen::Vector2d& angle, const Eigen::Vector2d& angular_velocities, const double l1, const double l2){
    Eigen::Matrix2d dJdt;
    dJdt(0,0) = -l1 * cos(angle(0))*angular_velocities(0) - l2 * cos(angle(0) + angle(1))*(angular_velocities(0)+angular_velocities(1));
    dJdt(0,1) = -l2 * cos(angle(0) + angle(1))*(angular_velocities(0)+angular_velocities(1));
    dJdt(1,0) = -l1 * sin(angle(0))*angular_velocities(0) - l2 * sin(angle(0) + angle(1))*(angular_velocities(0)+angular_velocities(1));
    dJdt(1,1) = -l2 * sin(angle(0) + angle(1))*(angular_velocities(0)+angular_velocities(1));
    return dJdt;
}

Eigen::Matrix2d inverse2d(const Eigen::Matrix2d & M){
    Eigen::Matrix2d M_inv;
    double det = M(0,0) * M(1,1) - M(0,1) * M(1,0);
    M_inv(0,0) = M(1,1);
    M_inv(0,1) = -M(0,1);
    M_inv(1,0) = -M(1,0);
    M_inv(1,1) = M(0,0);
    return M_inv / det;
}

Eigen::Vector2d transform_end_effector_position_to_joint_angles(const Eigen::Vector2d& end_effector_position, const double l1, const double l2, double tol){
    
    double c2{ (end_effector_position.squaredNorm() - l1*l1 - l2*l2)/(2.*l1*l2) };
    double q2{ acos(c2) }; // elbow angle
    double s2{ sin(q2) };
    double a{ l1 + l2*c2 };
    double b{ l2*s2 };
    double c1{ (a*end_effector_position(0) + b*end_effector_position(1)) / (a*a + b*b)};
    double s1{ (a*end_effector_position(1) - b*end_effector_position(0)) / (a*a + b*b)};
    double q1;
    if (s1 < 0) q1 = -acos(c1);
    else q1 = acos(c1);
    
    Eigen::Vector2d r;
    r << q1, q2;
    
    return r;
}

Eigen::Vector2d transform_joint_angles_to_end_effector_position(const Eigen::Vector2d& angles, const double l1, const double l2) {
    Eigen::Vector2d end_effector_position;
    end_effector_position <<    l1 * cos(angles(0)) + l2 * cos(angles(0) + angles(1)),
                            l1 * sin(angles(0)) + l2 * sin(angles(0) + angles(1));
    return end_effector_position;
}

Eigen::Vector2d transform_joint_angles_to_end_effector_position(const double angle_shoulder, const double angle_elbow, const double l1, const double l2) {
    Eigen::Vector2d end_effector_position;
    end_effector_position <<    l1 * cos(angle_shoulder) + l2 * cos(angle_shoulder + angle_elbow),
                            l1 * sin(angle_shoulder) + l2 * sin(angle_shoulder + angle_elbow);
    return end_effector_position;
}


Eigen::Vector2d transform_joint_velocity_to_end_effector_velocity(const Eigen::Vector2d& angles, const Eigen::Vector2d& angular_velocities, const double l1, const double l2) {
    Eigen::Vector2d end_effector_velocity;
    end_effector_velocity << -angular_velocities(0)*(l1*sin(angles(0)) + l2*sin(angles(0) + angles(1))) - angular_velocities(1)*l2*sin(angles(0) + angles(1)),
    angular_velocities(0)*(l1*cos(angles(0)) + l2*cos(angles(0) + angles(1))) + angular_velocities(1)*l2*cos(angles(0) + angles(1));
    
    return end_effector_velocity;
}

Eigen::Vector2d transform_joint_torque_to_end_effector_force(const Eigen::Vector2d& angles, const Eigen::Vector2d& torques, const double l1, const double l2) {
    
    Eigen::Matrix2d J = jacobian(angles, l1, l2);
    
    return inverse2d(J.transpose()) * torques;
}

void save_to_file(std::string folder, std::string type, RNN & net, Effector *imp, Input &inputs, int nb_reals){
    Monitor m_neuron, m_imp, m_input, m_state;
    int nb_targets = inputs.get_nb_targets();
    
    for (int i(0); i<nb_targets;i++){
        std::ofstream f_ctrl(folder+"control_"+type+"_"+std::to_string(i)+".txt");
        m_neuron.reset();
        m_imp.reset();
        m_input.reset();
        m_state.reset();
        
        for (int r(0);r<nb_reals;r++){
            Eigen::MatrixXd ctrls = net.forward(inputs, i, m_neuron, m_input, m_state);
            int duration = int(ctrls.cols());
            
            Eigen::VectorXd target_direction(imp->get_dim());
            target_direction << cos(2 * M_PI * i / double(nb_targets)), sin(2 * M_PI * i / double(nb_targets));
            Eigen::MatrixXd unit_target_directions(imp->get_dim(), duration);
            for (int c(0);c<duration;c++) unit_target_directions.col(c) = target_direction;
            
            // Output controls to file
            for (int col=0; col<duration; col++) f_ctrl << ctrls(0,col) << "\t" << ctrls(1,col) << std::endl;
    
            Eigen::MatrixXd states;
            if (imp->effector_type == DECODER){
                states = imp->get_end_effector_trajectory(m_neuron.get_data(duration));
            }
            else if(imp->effector_type == ARM){
                states = imp->get_end_effector_trajectory(ctrls);
            }
            else if (imp->effector_type == PASSIVECURSOR){
                states = imp->get_end_effector_trajectory(unit_target_directions);
            }
            m_imp.record_sequence(states);
        }
        m_state.save(folder+"state_"+type+"_"+std::to_string(i)+".txt");
        m_input.save(folder+"input_"+type+"_"+std::to_string(i)+".txt");
        m_neuron.save(folder+"activity_"+type+"_"+std::to_string(i)+".txt");
        m_imp.save(folder+"trajectory_"+type+"_"+std::to_string(i)+".txt");
        f_ctrl.close();
    }
}

void gather_data(RNN &net, Effector *imp, Monitor &m_neuron, Monitor &m_arm, Input &inputs, Target &targets, const int nb_reals){
    Eigen::MatrixXd ctrls, imp_states;
    m_arm.reset();
    m_neuron.reset();

    for (int r(0); r<nb_reals; r++){
        for (int i(0); i<targets.get_nb_targets();i++){
            ctrls = net.forward(inputs, i, m_neuron);
            int duration = int(ctrls.cols());
            Eigen::VectorXd target_direction = targets(i)/targets.get_distance_to_target();
            
            Eigen::MatrixXd unit_target_directions(imp->get_dim(), duration);
            for (int c(0);c<duration;c++) unit_target_directions.col(c) = target_direction;

            if (imp->effector_type == ARM) {
                imp_states = imp->get_end_effector_trajectory(ctrls);
            }
            else if (imp->effector_type == DECODER) {
                imp_states = imp->get_end_effector_trajectory(m_neuron.get_data(duration));
            }
            else if (imp->effector_type == PASSIVECURSOR){
                imp_states = imp->get_end_effector_trajectory(unit_target_directions);
            }
            Eigen::MatrixXd velocities = Eigen::MatrixXd::Zero(2, duration);
            velocities.row(0) = imp_states.row(2);
            velocities.row(1) = imp_states.row(3);
            m_arm.record_sequence(velocities);
        }
    }
}


double smoothstep (int edge0, int edge1, int x) {
    // Note : AMD implementation
   if (x < edge0)
      return 0;

   if (x >= edge1)
      return 1;

   // Scale/bias into [0..1] range
   double scaled_x = (x - edge0) / double(edge1 - edge0);

   return scaled_x * scaled_x * (3. - 2. * scaled_x);
}

std::tuple<Vector, Vector, Matrix> get_target_tuning_curves(const Matrix& targets, const Matrix& readout_activity, const int duration, const int hold_duration){
    const int nb_time_steps = int(readout_activity.cols());
    const int nb_units = int(readout_activity.rows());

    const int nb_trials = nb_time_steps / (duration + hold_duration);
    const int nb_targets = int(targets.rows()); //targets is nb_targets x dim
    const int dim = int(targets.cols());
    
    Vector b(nb_units), m(nb_units);
    Matrix P(nb_units, dim);
    
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
            mean_rates.row(i) += readout_activity.col((duration+hold_duration)*i+t).transpose();
        }
        mean_rates.row(i) /= double(duration/2);
    }
    
    // Solve
    Matrix beta = all_targets.colPivHouseholderQr().solve(mean_rates);
    b = beta.row(0).transpose();
    m = Vector::Zero(nb_units);
    for (int c=1; c<1+dim; c++) {
        m.array() += beta.row(c).transpose().array() * beta.row(c).transpose().array();
    }
    m.array() = sqrt(m.array());
    
    
    for (int r=0;r<P.rows();r++){
        for (int c=0; c<dim; c++){
            P(r, c) = beta(c+1, r) / m(r);
        }
    }
    return std::make_tuple(b, m, P);
}


Eigen::MatrixXd get_permutation_matrix(int size){
    Matrix P = Eigen::MatrixXd::Zero(size, size);
    std::vector<int> v(size);
    
    std::iota(v.begin(), v.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    
    for (int i=0;i<P.cols();i++){
        P(i, v[i]) = 1;
    }
    return P;
}

void output_eigen_matrix_to_file_stream(const Matrix& A, std::ofstream &f){
    // TODO: beware of column-major default with Eigen
    for (int r(0);r<A.rows();r++) {
        for (int c(0);c<A.cols()-1;c++) {
            f <<A(r, c)<< ",";
        }
        f <<A(r, A.cols()-1)<< std::endl;
    }
}

void output_eigen_matrix_to_file(const Matrix& A, const std::string &s){
    std::ofstream f(s);
    for (int r(0);r<A.rows();r++) {
        for (int c(0);c<A.cols()-1;c++) {
            f <<A(r, c)<< ",";
        }
        f <<A(r, A.cols()-1)<< std::endl;
    }
    f.close();
}

Vector variance(const Matrix& X, int ddof){
    const int n{ int(X.cols()) };
    Vector v(X.rows());
    v.setZero();
    Vector m = X.rowwise().mean();
    for (int c=0;c<X.cols();c++){
        v = v.array() + (X.col(c) - m).array()*(X.col(c) - m).array();
    }
    return v/(n - ddof);
}


Matrix lagged_autocovariance(const Matrix& X, const int lag, const int ddof){
    const long n_features{ X.rows() };
    const long n_samples{ X.cols() };
    
    // Subtract mean
    Matrix X_centered = X.colwise() - X.rowwise().mean();
    
    if (lag == 0){
        return (1./double(n_samples-ddof)) * X_centered * X_centered.transpose();
    }
    else {
        Matrix X_lagged1 = X_centered.block(0, 0, 0, n_samples-lag); // columns from 0 to n_samples-lag
        Matrix X_lagged2 = X_centered.block(0, lag, 0, n_samples-lag); // columns from lag to n_samples
        
        return (1./double(n_samples-lag-ddof)) * X_lagged1 * X_lagged2.transpose();
    }
}

bool contain(std::vector<size_t> v, size_t element){
    if (v.empty()) return false;
    else{
        bool in_v{ false };
        for (int i{ 0 }; i<v.size(); i++) {
            if (v[i] == element) {
                in_v = true;
                break;
            }
        }
        return in_v;
    }
}


