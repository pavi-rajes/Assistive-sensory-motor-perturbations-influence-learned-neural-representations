/**
 * Readout.hpp | bmi_model
 *
 * Description:
 * -----------
 * To read out the network activity to be fed to the BMI decoder.
 */

#ifndef Readout_hpp
#define Readout_hpp

#include <Eigen/Dense>
#include <vector>
#include <stdlib.h>

class Readout {
    Eigen::MatrixXd M;
    int n_channels;     // n_channels and n_units are identical and are the number of units to read from. 
    int n_units;
    int network_size;   // number of recurrent units in the RNN
    
public:
    ///Construct member matrix M so that n_units are singly recorded (i.e., one unit per channel).
    Readout(int _n_units, int network_size);
        
    ///Create Readout instance from `readout_matrix`
    Readout(const Eigen::MatrixXd & readout_matrix);
    
    ///Randomly create the readout matrix for a specified number of units and channels
    //Readout(int _n_channels, int _n_units, int network_size);
    
    ///Reading out an activity snapshot
    Eigen::VectorXd read(const Eigen::VectorXd& snap);
    
    ///Reading out a sequence of activities
    Eigen::MatrixXd read(const Eigen::MatrixXd& seq);
    
    ///Getters
    int get_nb_channels(){ return n_channels; }
    Eigen::MatrixXd get_readout_matrix(){ return M; }
    
    ///Set readout matrix
    void set_readout_matrix(const std::vector<size_t> & units);
    

};


#endif /* Readout_hpp */
