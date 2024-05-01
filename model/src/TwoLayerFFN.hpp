//  DEPRECATED!!! Kept for backward compatilibilty of some older code.
//  TwoLayerFFN.hpp
//  rnn4bci_cpp
//
//  Created by Alexandre Payeur on 2/1/22.
//

#ifndef TwoLayerFFN_hpp
#define TwoLayerFFN_hpp
#include <Eigen/Dense>
#include <tuple>
#include "DataGenerator.hpp"

class TwoLayerFFN {
    Eigen::MatrixXd Whi, Woh, gradWhi_cache, gradWoh_cache;
    Eigen::VectorXd bh, bo, gradbh_cache, gradbo_cache;
    Eigen::MatrixXd Ah_cache; // pre-activation cache for hidden layer
    int n_i, n_h, n_o;

    Eigen::MatrixXd backward(const Eigen::MatrixXd &Delta_out);    //returns hidden-layer errors
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> gradient(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Delta_h, const Eigen::MatrixXd &Delta_out);
    void update(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Delta_h, const Eigen::MatrixXd &Delta_out, double learning_rate_h, double learning_rate_o, double decay_rate);
    
public:
    TwoLayerFFN(int n_i, int n_h, int n_o);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &X);  //returns output
    std::pair<Eigen::VectorXd, Eigen::VectorXd> forward(const Eigen::VectorXd &x);
    void train(DataGenerator &datagen, int epochs, double learning_rate_h, double learning_rate_o=0, double decay_rate=0.9);
    
    static double MSEloss(Eigen::MatrixXd output, Eigen::MatrixXd target);
    
    ///Getters
    Eigen::MatrixXd get_Whi() {return Whi;}
    
    
};



#endif /* TwoLayerFFN_hpp */
