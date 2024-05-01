/**
 * utilities.hpp | bmi_model
 *
 * Description:
 * -----------
 * Mostly functions pertaining to arm transformation (todo: put as static members of TorqueBasedArm),
 * and data management.
 *
 */

#ifndef utilities_hpp
#define utilities_hpp

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "Monitor.hpp"
#include "RNN.hpp"
#include "VelocityKalmanFilter.hpp"

/// Functions dealing with transformations between end-effector and joint variables
Eigen::Vector2d condition_end_effector_in_joint_angles(const Eigen::Vector2d& angle, const Eigen::Vector2d& end_effector_position, const double l1, const double l2);
Eigen::Matrix2d jacobian(const Eigen::Vector2d& angle, const double l1, const double l2);
Eigen::Matrix2d djacobiandt(const Eigen::Vector2d& angle, const Eigen::Vector2d& angular_velocities, const double l1, const double l2);
Eigen::Matrix2d inverse2d(const Eigen::Matrix2d & M);
Eigen::Vector2d transform_end_effector_position_to_joint_angles(const Eigen::Vector2d& end_effector_position, const double l1, const double l2, double tol=1.e-8);
Eigen::Vector2d transform_joint_angles_to_end_effector_position(const Eigen::Vector2d& angles, const double l1, const double l2) ;
Eigen::Vector2d transform_joint_angles_to_end_effector_position(const double angle_shoulder, const double angle_elbow, const double l1, const double l2);
Eigen::Vector2d transform_joint_velocity_to_end_effector_velocity(const Eigen::Vector2d& angles, const Eigen::Vector2d& angular_velocities, const double l1, const double l2);
Eigen::Vector2d transform_joint_torque_to_end_effector_force(const Eigen::Vector2d& angles, const Eigen::Vector2d& torques, const double l1, const double l2);

/// Saving data to file
void save_to_file(std::string folder, std::string type, RNN & net, Effector *imp, Input &inputs, int nb_reals=1);

/// Gathering data for decoder fitting
void gather_data(RNN &net, Effector *imp, Monitor &m_neuron, Monitor &m_arm, Input &inputs, Target &targets, const int nb_reals=1);

/// Smooth step (smooth Heaviside function)
double smoothstep (int edge0, int edge1, int x);

/**
 * Compute target tuning curves from unit-normed targets and readout activities.
 *
 * Description:
 * Target tuning curves are computed by solving
 * r = b + m (p_x*d_x + p_y*d_y)    (e.g., when workspace in x-y plane)
 * for the bias (b), the modulation depth (m) and the unit-norm preferred direction (p_x, p_y).
 * d = (d_x, d_y) is a normalized target vector.
 *
 * Params:
 *  - targets: (nb_targets x dim), where dim = dimension of workspace (2, typically)
 *  - readout_activity: (nb_readout_units x (hold_duration+duration))
 *
 * Return:
 *  - tuple = <bias, modulation_depth, preferred_direction>
 *  where
 *  bias.size() = modulation_depth.size() = nb_readout_units
 *  preferred_direction is (nb_readout_units x dim)
 *
 */
std::tuple<Vector, Vector, Matrix> get_target_tuning_curves(const Matrix& targets, const Matrix& readout_activity,
                                                            const int duration, const int hold_duration);


Eigen::MatrixXd get_permutation_matrix(int size);

void output_eigen_matrix_to_file_stream(const Matrix& A, std::ofstream &f);
void output_eigen_matrix_to_file(const Matrix &A, const std::string &s);


/**
 * Compute var of matrix X along columns
 */
Vector variance(const Matrix& X, int ddof=0);


/**
 * Compute data covariance matrix for a given lag
 *
 * @param X : data matrix (n_features, n_samples)
 */
Matrix lagged_autocovariance(const Matrix& X, const int lag, const int ddof=0);


/**
 * Argsort function, stolen from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
 */
template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

/**
 * Function to determine if a vector<size_t> contains element i.
 */
bool contain(std::vector<size_t> v, size_t element);

/**
 * Returns all combinations of n elements taken k at a time, without replacement
 * Adapted from https://rosettacode.org/wiki/Combinations#C.2B.2B
 */
template <typename T>
std::vector< std::vector<T> > find_combinations(const std::vector<T> &v, const int k) {
    std::vector< std::vector<T> > combs;
    const int n{ int(v.size()) };

    std::string bitmask(k, 1);  // K leading 1's
    bitmask.resize(n, 0);       // N-K trailing 0's

    do {
       std::vector<T> comb;
       for (int i = 0; i < n; ++i) { // [0..n-1] integers
           if (bitmask[i]) comb.push_back(v[i]);
       }
       combs.push_back(comb);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return combs;
}


#endif /* utilities_hpp */
