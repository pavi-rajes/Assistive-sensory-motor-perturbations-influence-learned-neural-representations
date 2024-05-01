/**
 * Gradient.hpp | bmi_model
 *
 * Description:
 * -----------
 * Simple struct containing the gradients of the RNN.
 *
 * Note:
 * ----
 * Weirdly, the name of this file does not correspond to the name of the class...
 */
#ifndef Gradient_hpp
#define Gradient_hpp

#include <Eigen/Dense>
#include "EligibilityTrace.hpp"

struct GradientRecurrent {
    Eigen::MatrixXd U;
    Eigen::MatrixXd W;
    Eigen::MatrixXd V;
    Eigen::VectorXd b;
        
    GradientRecurrent(int Nin, int Nrec, int Nout);
    GradientRecurrent(const GradientRecurrent& g);
    GradientRecurrent& operator+=(const GradientRecurrent& g);
    GradientRecurrent& operator=(const GradientRecurrent& g);

    
    void reset();
    void update(double delta_reward, const EligibilityTraceRecurrent& et);
    void update(double reward, double reward_trace, const EligibilityTraceRecurrent& et);
};

#endif /* Gradient_hpp */
