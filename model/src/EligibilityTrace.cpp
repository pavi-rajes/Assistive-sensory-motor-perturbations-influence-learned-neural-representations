/**
 * EligibilityTrace.cpp | bmi_model
 */

#include "EligibilityTrace.hpp"

EligibilityTraceRecurrent::EligibilityTraceRecurrent(int Nin, int Nrec, int Nout) {
    U = Eigen::MatrixXd::Zero(Nrec, Nin);
    W = Eigen::MatrixXd::Zero(Nrec, Nrec);
    V = Eigen::MatrixXd::Zero(Nout, Nrec);
    b = Eigen::VectorXd::Zero(Nrec);
}

void EligibilityTraceRecurrent::reset() {
    U.setZero();
    W.setZero();
    V.setZero();
    b.setZero();
}
