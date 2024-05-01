//  DEPRECATED!!! Kept for backward compatilibilty of some older code.
//  TwoLayerFFN.cpp
//  rnn4bci_cpp
//
//  Created by Alexandre Payeur on 2/1/22.
//

#include "TwoLayerFFN.hpp"
#include "rand_mat.hpp"
#include <iostream>
#include <fstream>

TwoLayerFFN::TwoLayerFFN(int n_i, int n_h, int n_o) : n_i(n_i), n_h(n_h), n_o(n_o){
    bh = Eigen::VectorXd::Zero(n_h);
    bo = Eigen::VectorXd::Zero(n_o);
    Whi = random_gaussian_matrix(n_h, n_i, 1./sqrt(n_h));
    Woh = random_gaussian_matrix(n_o, n_h, 1./sqrt(n_h));
    
    gradbh_cache = Eigen::VectorXd::Zero(n_h);
    gradbo_cache = Eigen::VectorXd::Zero(n_o);
    gradWhi_cache = Eigen::MatrixXd::Zero(n_h, n_i);
    gradWoh_cache = Eigen::MatrixXd::Zero(n_o, n_h);
}

Eigen::MatrixXd TwoLayerFFN::forward(const Eigen::MatrixXd &X){
    Ah_cache = Whi * X;  // X is (n_i x # examples)
    Ah_cache.colwise() += bh;
    Eigen::MatrixXd H = Eigen::tanh(Ah_cache.array());
    Eigen::MatrixXd O = Woh * H;
    O.colwise() += bo;
    return O ;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> TwoLayerFFN::forward(const Eigen::VectorXd &x){
    Eigen::VectorXd h = Whi * x + bh;
    h = Eigen::tanh(h.array());
    return std::make_pair(h, Woh * h + bo);
}

Eigen::MatrixXd TwoLayerFFN::backward(const Eigen::MatrixXd &Delta_out){
    Eigen::MatrixXd weight_prop_error = Woh.transpose() * Delta_out;  //n_h x # examples
    Eigen::MatrixXd Delta_h = (1 - Eigen::tanh(Ah_cache.array()) * Eigen::tanh(Ah_cache.array())) * weight_prop_error.array();
    return Delta_h;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> TwoLayerFFN::gradient(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Delta_h, const Eigen::MatrixXd &Delta_out){
    
    Eigen::VectorXd dbh(n_h);
    Eigen::VectorXd dbo(n_o);
    Eigen::MatrixXd dWhi(n_h, n_i);
    Eigen::MatrixXd dWoh(n_o, n_h);

    dbh = Delta_h.rowwise().mean();
    dbo = Delta_out.rowwise().mean();
    dWhi = Delta_h * X.transpose() / X.cols();
    dWoh = Delta_out * (Eigen::tanh(Ah_cache.array())).matrix().transpose() / X.cols();
    
    return std::make_tuple(dWhi, dbh, dWoh, dbo);
}

void TwoLayerFFN::update(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Delta_h, const Eigen::MatrixXd &Delta_out, double learning_rate_h, double learning_rate_o, double decay_rate){
    //RMSprop
    //cache = decay_rate * cache + (1 - decay_rate) * dx**2
    //x += - learning_rate * dx / (np.sqrt(cache) + eps)
    auto grads = gradient(X, Delta_h, Delta_out);
    gradWhi_cache = decay_rate * gradWhi_cache.array() + (1 - decay_rate) * std::get<0>(grads).array() * std::get<0>(grads).array();
    gradbh_cache = decay_rate * gradbh_cache.array() + (1 - decay_rate) * std::get<1>(grads).array() * std::get<1>(grads).array();
    gradWoh_cache = decay_rate * gradWoh_cache.array() + (1 - decay_rate) * std::get<2>(grads).array() * std::get<2>(grads).array();
    gradbo_cache = decay_rate * gradbo_cache.array() + (1 - decay_rate) * std::get<3>(grads).array() * std::get<3>(grads).array();
    
    Whi = Whi - learning_rate_h * (std::get<0>(grads).array() / (Eigen::sqrt(gradWhi_cache.array()) + 1e-6)).matrix();
    bh = bh - learning_rate_h * (std::get<1>(grads).array() / (Eigen::sqrt(gradbh_cache.array()) + 1e-6)).matrix();
    Woh = Woh - learning_rate_o * (std::get<2>(grads).array() / (Eigen::sqrt(gradWoh_cache.array()) + 1e-6)).matrix();
    bo = bo - learning_rate_o * (std::get<3>(grads).array() / (Eigen::sqrt(gradbo_cache.array()) + 1e-6)).matrix();
}


void TwoLayerFFN::train(DataGenerator &datagen, int epochs, double learning_rate_h, double learning_rate_o, double decay_rate){
    std::ofstream f("loss_ffn.txt");

    // Generate training data
    auto training_data = datagen.generate(60000);
    Eigen::MatrixXd X = std::get<0>(training_data);
    
    // Generate test data
    auto test_data = datagen.generate(10000);
    
    for (int e=0;e<epochs;e++){
        // Forward
        Eigen::MatrixXd output = forward(std::get<0>(training_data));
        
        // Backward
        Eigen::MatrixXd delta_out = output - std::get<1>(training_data);
        Eigen::MatrixXd delta_h = backward(delta_out);
        
        // Update weights and biases
        update(std::get<0>(training_data), delta_h, delta_out, learning_rate_h, learning_rate_o, decay_rate);
        
        double training_loss = MSEloss(output, std::get<1>(training_data));
        
        output = forward(std::get<0>(test_data));
        double test_loss = MSEloss(output, std::get<1>(test_data));
        
        // Print out
        std::cout << "Epoch " << e + 1 << ": Training = " << training_loss << " | Test = " << test_loss << std::endl;
        f << e + 1 << "\t" << training_loss << "\t" << test_loss << std::endl;
    }
    f.close();
}


double TwoLayerFFN::MSEloss(Eigen::MatrixXd output, Eigen::MatrixXd target){
    double n_examples = double(output.cols());
    Eigen::MatrixXd tmp = (output.array() - target.array()) * (output.array() - target.array());
    return (0.5/n_examples) * tmp.sum();
}
