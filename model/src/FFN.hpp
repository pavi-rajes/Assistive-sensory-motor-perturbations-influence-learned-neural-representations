/**
 * FFN.hpp | bmi_model
 *
 * Description:
 * -----------
 * Feedforward neural network trained using backprop.
 * In the context of the model, the FFN is used to encode information
 * about the environment (e.g., target position, effector position).
 *
 * Notes:
 * -----
 * 1. By default, the output layer activation function is the identity function.
 * 2. The only optimizer implemented is Adam.
 */

#ifndef FFN_hpp
#define FFN_hpp

#include <vector>
#include "DataGenerator.hpp"
#include "globals.hpp"
#include "activations.hpp"

class FFN {
    std::vector<unsigned int> topology;         // [n_input, n_hidden1, n_hidden2, ..., n_out]
    int nb_layers;                              // by definition, number of weight matrices
    int nb_unit_layers;                         // total number of layers, counting the input layer, all hidden layers and the output layer
    int nb_hidden_unit_layers;                  // = nb_layers - 1 = nb_unit_layers - 2
    
    std::vector<Vector> preactivations;         // hidden layer preactivation
    std::vector<Matrix> preactivations_mat;
    
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    
    std::vector<Matrix> grad_w_accumulator;     // Accumulates gradients until update is performed
    std::vector<Vector> grad_b_accumulator;
    
    std::vector<Matrix> m;                      // 1st order moments (for Adam optimization)
    std::vector<Vector> m_b;

    std::vector<Matrix> v;                      // 2nd order moments (for Adam optimization)
    std::vector<Vector> v_b;
    
    DataGenerator::DataTypeFFN input_type;      // Data generator for training and testing
    
    ActivationType phi;
    
    void backward(const Vector &x, const Vector& Delta_out);  // backward error propagation
    void backward(const Matrix &X, const Matrix& Delta_out);
    void update(int batch_size, const std::vector<double>& lr, double decay_rate); // updating Adam's moments, gradients and resetting gradient accumulators
    
public:
    /// Constructor
    FFN(std::vector<unsigned int> layer_sizes, ActivationType phi=ActivationType::tanh);
    
    /// Forward propagation
    Matrix forward(const Matrix& x);
    Vector forward(const Vector& x);
    
    /// Getting the penultimate hidden layer activation to use as input to the RNN
    /// If the penultimate hidden layer does not exist (1-hidden-layer FFN), then the single hidden layer output is used.
    Vector penultimate_hidden_layer_activation(const Vector &x);
    
    /// Train
    void train(DataGenerator &datagen, int epochs, double lr_hidden, double lr_out=0, double decay_rate=0.9, int batch_size=50);
    
    /// Setter
    void set_input_type(DataGenerator::DataTypeFFN dt) {input_type = dt;}
    
    /// Getters
    int get_output_size_to_RNN();
    int get_input_size();
    Matrix get_U();
    DataGenerator::DataTypeFFN get_input_type() { return input_type; }
    
    /// Saving to file and loading from file (SUBOPTIMAL: assumes that networks are compatible in size)
    void save(std::string file_prefix);
    void load(std::string file_prefix);
};



#endif /* FFN_hpp */
