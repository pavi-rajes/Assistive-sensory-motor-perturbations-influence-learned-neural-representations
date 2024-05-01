/**
 * EligibilityTrace.hpp | bmi_model
 *
 * Description:
 * -----------
 * Simple struct that keeps the eligibility traces in memory, for each matrix of the recurrent network.
 */
#ifndef EligibilityTrace_hpp
#define EligibilityTrace_hpp

#include <Eigen/Dense>

struct EligibilityTraceRecurrent {
    Eigen::MatrixXd U;
    Eigen::MatrixXd W;
    Eigen::MatrixXd V;
    Eigen::VectorXd b;
    
    EligibilityTraceRecurrent(int Nin, int Nrec, int Nout);
    void reset();
};

#endif /* EligibilityTrace_hpp */
