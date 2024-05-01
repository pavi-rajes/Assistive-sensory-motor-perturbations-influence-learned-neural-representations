/**
 * Gradient.cpp | bmi_model
 */

#include "Gradient.hpp"
#include <iostream>
GradientRecurrent::GradientRecurrent(int Nin, int Nrec, int Nout){
    U = Eigen::MatrixXd::Zero(Nrec, Nin);
    W = Eigen::MatrixXd::Zero(Nrec, Nrec);
    V = Eigen::MatrixXd::Zero(Nout, Nrec);
    b = Eigen::VectorXd::Zero(Nrec);
}

GradientRecurrent::GradientRecurrent(const GradientRecurrent& g){
    this->U = g.U;
    this->W = g.W;
    this->V = g.V;
    this->b = g.b;
}

GradientRecurrent& GradientRecurrent::operator+=(const GradientRecurrent& g){
    this->U += g.U;
    this->W += g.W;
    this->V += g.V;
    this->b += g.b;
    return *this;
}

GradientRecurrent& GradientRecurrent::operator=(const GradientRecurrent& g){
    this->U = g.U;
    this->W = g.W;
    this->V = g.V;
    this->b = g.b;
    return *this;
}

void GradientRecurrent::reset() {
    U.setZero();
    W.setZero();
    V.setZero();
    b.setZero();
}

void GradientRecurrent::update(double delta_reward, const EligibilityTraceRecurrent& et){
    U += delta_reward * et.U;
    W += delta_reward * et.W;
    b += delta_reward * et.b;
}

void GradientRecurrent::update(double reward, double reward_trace, const EligibilityTraceRecurrent& et){
    U += (reward - reward_trace) * et.U;
    W += (reward - reward_trace) * et.W;
    b += (reward - reward_trace) * et.b;
    //std::cout << (reward - reward_trace) * et.W << std::endl;
}
