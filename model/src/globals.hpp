/**
 * globals.hpp | bmi_model
 *
 * Description:
 * -----------
 * Some global variables and aliases are defined.
 */
#ifndef globals_hpp
#define globals_hpp

#include <Eigen/Dense>

extern unsigned seed_matrix;    //seed for random matrices
extern const double dt;         //integration time step

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

#endif 
