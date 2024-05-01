/**
 * rand_mat.hpp | bmi_model
 *
 * Definitions of functions returning random vectors and matrices.
 */

#include "rand_mat.hpp"
#include "globals.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

std::mt19937 myrng;
std::normal_distribution<double> Normal(0.0, 1.0);
std::uniform_real_distribution<double> Uniform(0.0, 1.0);


Eigen::MatrixXd random_gaussian_matrix(const int nrows, const int ncols, const double scaling) {
    Eigen::MatrixXd G(nrows, ncols);
    for (int rr=0; rr < G.rows(); rr++)
        for (int cc=0; cc < G.cols(); cc++) {
            G(rr, cc) =  scaling * Normal(myrng);
        }
    return G;
}

Eigen::MatrixXd random_uniform_matrix(const int nrows, const int ncols, const double scaling){
    Eigen::MatrixXd G(nrows, ncols);
    for (int rr=0; rr < G.rows(); rr++)
        for (int cc=0; cc < G.cols(); cc++) {
            G(rr, cc) =  scaling * (2 * Uniform(myrng) - 1);
        }
    return G;
}

Eigen::MatrixXd random_uniform_matrix(const int nrows, const int ncols, const double low, const double high){
    Eigen::MatrixXd G(nrows, ncols);
    for (int rr=0; rr < G.rows(); rr++)
        for (int cc=0; cc < G.cols(); cc++) {
            G(rr, cc) =  low + (high - low) * Uniform(myrng);;
        }
    return G;
}

Eigen::MatrixXd balanced_uniform_matrix(const int nrows, const int ncols, const double scaling){
    Eigen::MatrixXd V = random_uniform_matrix(nrows, ncols, scaling);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(ncols);
    Eigen::MatrixXd Ones = Eigen::MatrixXd::Ones(ncols, ncols);
    double lr{ 0.1 };
    /*Eigen::MatrixXd G(nrows, ncols);
    for (int rr=0; rr < G.rows(); rr++){
        std::vector<int> v(ncols);
        std::iota(v.begin(), v.end(), 0);
        std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
        for (int cc=0; cc < ncols/2; cc++) G(rr, v[cc]) =  scaling * Uniform(myrng);
        for (int cc=ncols/2; cc<ncols; cc++) G(rr, v[cc]) =  -scaling * Uniform(myrng);
    }*/
    
    while ((V * ones).norm() > 1e-10 ){
        V = V - lr * V * Ones;
    }
    
    return V;
}

Eigen::MatrixXd balanced_gaussian_matrix(const int nrows, const int ncols, const double scaling){
    Eigen::MatrixXd V = random_gaussian_matrix(nrows, ncols, scaling);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(ncols);
    Eigen::MatrixXd Ones = Eigen::MatrixXd::Ones(ncols, ncols);
    double lr{ 0.001 };
    
    std::cout << "Initial V\n";
    std::cout << V << std::endl;
    while ((V * ones).norm() > 1e-10 ){
        V = V - lr * V * Ones;
    }
    std::cout << "Final V\n";
    std::cout << V << std::endl;
    return V;
}



Eigen::VectorXd random_gaussian_vector(const int size){
    Eigen::VectorXd v(size);
    for (int nn = 0; nn < v.size(); nn++)
        v(nn) = Normal(myrng);
    return v;
}

Eigen::VectorXd random_uniform_vector(const int size, const double low, const double high){
    Eigen::VectorXd v(size);
    for (int nn = 0; nn < v.size(); nn++)
        v(nn) = low + (high - low) * Uniform(myrng);
    return v;
}
