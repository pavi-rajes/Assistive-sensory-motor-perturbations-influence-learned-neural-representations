/**
 * FFN.cpp | bmi_model
 */

#include "FFN.hpp"
#include "rand_mat.hpp"
#include <fstream>
#include "TwoLayerFFN.hpp"

FFN::FFN(std::vector<unsigned int> topology, ActivationType phi) : phi(phi){
    //input -> W0 -> h[0] -> W[1] -> h[1] -> ... W[L-1] -> h[L-1] -> W[L] -> output
    input_type = DataGenerator::unspecified;
    
    nb_unit_layers = int(topology.size());
    nb_layers = nb_unit_layers - 1;
    nb_hidden_unit_layers = nb_layers - 1;
    
    this->topology = topology;
    
    preactivations.reserve(nb_hidden_unit_layers); // only for hidden units
    weights.reserve(nb_layers);
    biases.reserve(nb_layers);
    grad_w_accumulator.reserve(nb_layers);
    grad_b_accumulator.reserve(nb_layers);
    m.reserve(nb_layers);
    m_b.reserve(nb_layers);
    v.reserve(nb_layers);
    v_b.reserve(nb_layers);

    
    for (int i=0;i<nb_hidden_unit_layers;i++) preactivations.push_back(Vector::Zero(topology[i+1]));
    
    if (phi == ActivationType::relu){
        for (int i=0;i<nb_layers;i++) biases.push_back(random_uniform_vector(topology[i+1], 0.5, 1.5));
        //for (int i=0;i<nb_layers;i++) biases.push_back(Vector::Zero(topology[i+1]));
    }
    else if (phi == ActivationType::retanh) {
        for (int i=0;i<nb_layers;i++) biases.push_back(0.5*Vector::Ones(topology[i+1]));
    }
    else {
        for (int i=0;i<nb_layers;i++) biases.push_back(Vector::Zero(topology[i+1]));
    }
    
    
    for (int i=0;i<nb_layers;i++) grad_b_accumulator.push_back(Vector::Zero(topology[i+1]));
    
    for (int i=0;i<nb_layers;i++) m_b.push_back(Vector::Zero(topology[i+1]));
    
    for (int i=0;i<nb_layers;i++) v_b.push_back(Vector::Zero(topology[i+1]));
    
    
    if (phi == ActivationType::relu or phi == ActivationType::retanh){
        for (int i=0;i<nb_layers;i++) weights.push_back(random_gaussian_matrix(topology[i+1], topology[i], sqrt(2./topology[i])));
    } else {
        for (int i=0;i<nb_layers;i++) weights.push_back(random_gaussian_matrix(topology[i+1], topology[i], sqrt(2./(topology[i+1]+topology[i]))));
    }
    
    for (int i=0;i<nb_layers;i++) grad_w_accumulator.push_back(Matrix::Zero(topology[i+1], topology[i]));
    
    for (int i=0;i<nb_layers;i++) m.push_back(Matrix::Zero(topology[i+1], topology[i]));
    
    for (int i=0;i<nb_layers;i++) v.push_back(Matrix::Zero(topology[i+1], topology[i]));
    
}

Matrix FFN::forward(const Matrix &X){
    Matrix H = X;
    for (int layer=0;layer<nb_layers-1;layer++){
        H = weights[layer] * H;
        H.colwise() += biases[layer];
        if (layer < nb_hidden_unit_layers and preactivations_mat.size()>0) preactivations_mat[layer] = H;
        H = activate(H, phi);
    }
    H = weights[nb_layers-1] * H;  //linear output units for linear decoding
    H.colwise() += biases[nb_layers-1];
    
    return H ;
}

Vector FFN::forward(const Vector &x){
    Vector h = x;
    for (int layer=0;layer<nb_layers-1;layer++){
        h = weights[layer] * h;
        h.colwise() += biases[layer];
        if (layer < nb_hidden_unit_layers) preactivations[layer] = h;
        h = activate(h, phi);
    }
    h = weights[nb_layers-1] * h;  //linear output units for linear decoding
    h.colwise() += biases[nb_layers-1];
    
    return h ;
}

Vector FFN::penultimate_hidden_layer_activation(const Vector &x){
    Vector h = x;
    if (nb_hidden_unit_layers > 1){
        for (int layer=0;layer<nb_hidden_unit_layers-1;layer++){
            h = weights[layer] * h;
            h.colwise() += biases[layer];
            h = activate(h, phi);
        }
    }
    else {
        h = weights[0] * h;
        h.colwise() += biases[0];
        h = activate(h, phi);
    }
    return h ;
}

void FFN::backward(const Vector &x, const Vector& delta_out){
    Vector d = delta_out;
    for (int layer=nb_layers-1;layer>=1;layer--){
        Vector a = activate(preactivations[layer-1], phi);
        grad_w_accumulator[layer] += d * a.transpose();
        grad_b_accumulator[layer] += d;
        
        d = weights[layer].transpose() * d;
        
        d = derivate(a, phi).array() * d.array();
    }
    grad_w_accumulator[0] += d * x.transpose();
    grad_b_accumulator[0] += d;
}

void FFN::backward(const Matrix &X, const Matrix& Delta_out){
    Matrix D = Delta_out;
    for (int layer=nb_layers-1;layer>=1;layer--){
        Matrix A = activate(preactivations_mat[layer-1], phi);
        grad_w_accumulator[layer] = D * A.transpose();
        grad_b_accumulator[layer] = D.rowwise().sum();
        
        D = weights[layer].transpose() * D;
        D = derivate(A, phi).array() * D.array();
    }
    grad_w_accumulator[0] = D * X.transpose();
    grad_b_accumulator[0] = D.rowwise().sum();

}

void FFN::update(int batch_size, const std::vector<double>& lr, double decay_rate){
    
    for (int layer=nb_layers-1;layer>=0;layer--){
        grad_w_accumulator[layer] /= double(batch_size);
        grad_b_accumulator[layer] /= double(batch_size);
        
        m[layer] = 0.9 * m[layer] + (1 - 0.9) * grad_w_accumulator[layer];
        m_b[layer] = 0.9 * m_b[layer] + (1 - 0.9) * grad_b_accumulator[layer];
        
        v[layer] = decay_rate * v[layer].array() + (1 - decay_rate) * grad_w_accumulator[layer].array() * grad_w_accumulator[layer].array();
        v_b[layer] = decay_rate * v_b[layer].array() + (1 - decay_rate) * grad_b_accumulator[layer].array() * grad_b_accumulator[layer].array();

        weights[layer] += - lr[layer] * (m[layer].array() / (Eigen::sqrt(v[layer].array()) + 1e-8)).matrix();
        biases[layer] += - lr[layer] * (m_b[layer].array() / (Eigen::sqrt(v_b[layer].array()) + 1e-8)).matrix();
        
        grad_w_accumulator[layer].setZero();
        grad_b_accumulator[layer].setZero();
    }
}


void FFN::train(DataGenerator &datagen, int epochs, double lr_hidden,double lr_out, double decay_rate, int batch_size){
    std::ofstream f("loss_ffn.txt");
    input_type = datagen.get_datatype();
        
    // Number of batches
    constexpr int nb_training_ex(50000);  //TODO: don't hard-code nb of training examples
    const int nb_batches = nb_training_ex/batch_size;
    
    // Preactivation buffer
    for (int i=0;i<nb_hidden_unit_layers;i++) preactivations_mat.push_back(Matrix::Zero(topology[i+1], batch_size));
    
    // Learning rates
    std::vector<double> lr(nb_layers);
    std::vector<double> lr_zero(nb_layers);
    lr[nb_layers - 1] = lr_out;
    for (int i=0;i<nb_layers-1;i++) lr[i] = lr_hidden;
    for (int i=0;i<nb_layers;i++) lr_zero[i] = 0.;

    // Training
    for (int e=0;e<epochs;e++){
        double training_loss{ 0. };
        for (int batch=0;batch<nb_batches;batch++){
            
            // Generate training data for a batch
            auto training_data = datagen.generate(batch_size);
            Matrix X = std::get<0>(training_data);
            Matrix O = std::get<1>(training_data);
            
            // Forward
            Matrix Y = forward(X);
            
            // Backward
            Matrix Delta_out = Y.array() - O.array();
            backward(X, Delta_out);
            
            /*for (int ex_id=0;ex_id<batch_size;ex_id++){
                // Forward
                Vector input = X.col(ex_id); //X.col(*iter);
                Vector output = forward(input);
                
                // Backward
                Vector delta_out = output - O.col(ex_id); //output - O.col(*iter);
                backward(input, delta_out);
            }*/
             
            // Update weights and biases
            if (e == 0) update(batch_size, lr_zero, decay_rate);  // allows relaxation of elementwise squared grad
            else update(batch_size, lr, decay_rate);
            
            // Batch loss
            Y = forward(X);
            training_loss += batch_size*TwoLayerFFN::MSEloss(Y, O);
        }
        training_loss /= nb_training_ex;
        
        // Generate test data
        auto test_data = datagen.generate(nb_training_ex / 10);
        Matrix Y = forward(std::get<0>(test_data));
        double test_loss = TwoLayerFFN::MSEloss(Y, std::get<1>(test_data));
        
        // Print out
        if (e % (epochs/10) == 0) std::cout << "Epoch " << e + 1 << ": Training = " << training_loss << " | Test = " << test_loss << std::endl;
        f << e + 1 << "\t" << training_loss << "\t" << test_loss << std::endl;
        
    }
    f.close();
}


int FFN::get_output_size_to_RNN(){
    if (nb_hidden_unit_layers > 1) return topology[nb_unit_layers-3];
    else return topology[1];
}

int FFN::get_input_size() {
    return topology[0];
}

Matrix FFN::get_U(){
    return weights[nb_layers-2];
}

void FFN::save(std::string file_prefix){
    std::ofstream f;
    for (int i=0; i<nb_layers; i++){
        // Biases
        f.open(file_prefix + "_b" + std::to_string(i) + ".txt");
        for (int r(0);r<biases[i].size()-1;r++){
            f << biases[i].data()[r] << ",";
        }
        f << biases[i](biases[i].size()-1) << std::endl;
        f.close();
        // Weights
        f.open(file_prefix + "_W" + std::to_string(i) + ".txt");
        for (int r(0);r<weights[i].rows();r++) {
            for (int c(0);c<weights[i].cols()-1;c++) {
                f <<weights[i](r, c)<< ",";
            }
            f <<weights[i](r, weights[i].cols()-1) << std::endl;
        }
        f.close();
    }
}

void FFN::load(std::string file_prefix){
    std::ifstream f;
    std::string line;
    double val;
    int row_counter, col_counter;
    
    for (int i=0; i<nb_layers; i++){
        // Biases
        f.open(file_prefix + "_b" + std::to_string(i) + ".txt");
        if(!f.is_open()) throw std::runtime_error("Could not open file: "+file_prefix + "_b" + std::to_string(i) + ".txt");
        row_counter=0;
        while (std::getline(f, line)){
            std::stringstream ss(line);
            col_counter = 0;
            while (ss >> val){
                biases[i](col_counter) = val;
                if(ss.peek() == ',') ss.ignore();
                col_counter++;
            }
            row_counter++;
        }
        f.close();
        
        // Weights
        f.open(file_prefix + "_W" + std::to_string(i) + ".txt");
        if(!f.is_open()) throw std::runtime_error("Could not open file");
        
        row_counter = 0;
        while (std::getline(f, line)){
            std::stringstream ss(line);
            col_counter = 0;
            while (ss >> val){
                weights[i](row_counter, col_counter) = val;
                if(ss.peek() == ',') ss.ignore();
                col_counter++;
            }
            row_counter++;
        }
        f.close();
    }
}
