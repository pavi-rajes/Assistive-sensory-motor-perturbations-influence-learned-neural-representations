/**
 * rand_mat.hpp | bmi_model
 *
 * Description:
 * -----------
 * Declarations of functions returning random vectors and matrices.
 */

#ifndef rand_mat_hpp
#define rand_mat_hpp

#include <Eigen/Dense>
#include <random>

extern std::mt19937 myrng;  //not very clean... try to avoid using a global variable here
Eigen::MatrixXd random_gaussian_matrix(const int nrows, const int ncols, const double scaling);
Eigen::MatrixXd random_uniform_matrix(const int nrows, const int ncols, const double scaling);
Eigen::MatrixXd random_uniform_matrix(const int nrows, const int ncols, const double low, const double high);
Eigen::VectorXd random_gaussian_vector(const int size);
Eigen::VectorXd random_uniform_vector(const int size, const double low=0., const double high=1.);
Eigen::MatrixXd balanced_uniform_matrix(const int nrows, const int ncols, const double scaling);
Eigen::MatrixXd balanced_gaussian_matrix(const int nrows, const int ncols, const double scaling);

#endif /* rand_mat_hpp */
