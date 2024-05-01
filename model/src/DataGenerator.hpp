/**
 * DataGenerator.hpp | bmi_model
 *
 * Classes to generate training and test data to be encoded by the FFN used as input to the RNN.
 * IMPORTANT NOTE: Only a 2D workspace (i.e., dim = 2) is supported at the moment.
 *
 * Contains 4 classes:
 *  1) DataGenerator    :   Abstract base class
 *
 *  2) TargetGenerator  :   Each input represents the (x,y)-position of a target and the context vector [(0,1) or (1,0)].
 *                          Each output represents the (x,y)-position of a target.
 *                          The goal is to encode the position of the target under manual and brain control.
 *
 *                          input = [target_position_x; target_position_y; context_vector]
 *                          ouput = [target_position_x; target_position_y]
 *
 *  3) MixedGenerator   :   Each input represents the (x,y)-position of the end-effector, the (x,y)-position of the target and the context vector.
 *                          Each output represents the (x,y)-position of the end-effector, the (x,y)-position of the target as well as the relative position of the target with respect to the end-effector.
 *
 *                          input = [effector_position_x; effect_position_y;
 *                                   target_position_x; target_position_y;
 *                                   context_vector]
 *                          output = [effector_position_x; effect_position_y;
 *                                   target_position_x; target_position_y;
 *                                   target_position_x - effector_position_x; target_position_y - effector_position_y]
 *
 *  4) TorqueBasedArmGenerator  :   Rarely used. Represents both proprioceptive (e.g., angular variables)
 *                                  and visual (e.g., hand/cursor position) input/output pairs.
 */
#ifndef DataGenerator_hpp
#define DataGenerator_hpp

#include <utility>
#include <Eigen/Dense>
#include <random>
#include "TorqueBasedArm.hpp"

/// Abstract class for data generation.
class DataGenerator {
public:
    enum DataTypeFFN {
        unspecified,
        target_only,
        target_and_position,
        torque_based_arm,
    };
    /**
     * Generate n samples
     *
     * @param n     Number of samples to generate
     * @return      <input, output> pairs of samples placed in the columns of the matrices
     *
     */
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generate(int n) =0;
    virtual ~DataGenerator() {}
    virtual int get_input_dim() =0;
    virtual int get_output_dim() =0;
    DataTypeFFN get_datatype() { return datatype; }
    
protected:
    DataGenerator::DataTypeFFN datatype = {unspecified};
};


class TargetGenerator:public DataGenerator {
protected:
    int dim;
    double max_radius;
    std::mt19937 rng;
    
public:
    TargetGenerator(int dim, double r);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generate(int n);
    static std::vector<int> find_context(const Eigen::MatrixXd &X, short int c);
    int get_input_dim();
    int get_output_dim();
};


class MixedGenerator:public DataGenerator {
protected:
    int dim;
    double max_radius;
    std::mt19937 rng;
    
public:
    MixedGenerator(int dim, double r);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generate(int n);
    int get_input_dim();
    int get_output_dim();
};


class TorqueBasedArmGenerator : public DataGenerator {
protected:
    int dim;
    double max_radius;
    double distance_to_target;
    TorqueBasedArm arm;
    std::mt19937 rng;
public:
    /**
     * Ctor
     * radius and distance_to_target in meters, contrary to FFNInput (TODO: standardize this)
     */
    TorqueBasedArmGenerator(int dim, double max_radius, double distance_to_target, const TorqueBasedArm& arm);
    
    /// Genrate random data for TorqueBasedArm with proprioception
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generate(int n);
    
    /// Get input and output dimensions (typically used before training the FFN itself)
    int get_input_dim();
    int get_output_dim();
};

#endif /* DataGenerator_hpp */
