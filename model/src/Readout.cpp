/**
 * Readout.cpp | bmi_model
 */
#include "Readout.hpp"
#include <iostream>

Readout::Readout(int _n_units, int network_size):n_units(_n_units), network_size(network_size){
    n_channels = _n_units;
    M = Eigen::MatrixXd::Identity(n_channels, network_size);
}

Readout::Readout(const Eigen::MatrixXd & readout_matrix){
    M = readout_matrix;
    n_channels = int(M.rows());
    n_units = int(M.cols());
}

/*Readout::Readout(int _n_channels, int _n_units, int network_size):n_units(_n_units), n_channels(_n_channels){
    if (n_units < n_channels) {
        std::cerr << "Number of units < number of channels. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        M = Eigen::MatrixXi::Zero(n_channels, network_size);
        int n_units_per_channel = n_units / n_channels;
        int remainder = n_units % n_channels;
        
        for
        
        
        if (n_channels == n_units) {
            for (int i=0;i<n_units;i++) M(i, i) = 1;  //we can select the first n_units without loss of generality
        } else {
            int n_units_per_channel = n_units / n_channels;
            if (n_units_per_channel == 1) {
                for (int j=0;j<remainder)
            }
        }
    }
}*/

Eigen::VectorXd Readout::read(const Eigen::VectorXd& snap){
    return M * snap;
}

Eigen::MatrixXd Readout::read(const Eigen::MatrixXd& seq){
    return M * seq;
}

void Readout::set_readout_matrix(const std::vector<size_t> & units){
    n_channels = int(units.size());
    n_units = n_channels;  //DEBUG: change if at some point n_units != n_channels
    M = Eigen::MatrixXd::Zero(n_channels, network_size);
    for (int i(0);i<units.size();i++) M(i, units[i]) = 1.;
}
