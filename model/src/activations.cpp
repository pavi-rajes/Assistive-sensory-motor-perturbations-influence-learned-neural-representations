/**
 * activations.cpp | bmi_model
 */

#include "activations.hpp"

Vector activate(const Vector &x, ActivationType phi){
    switch (phi) {
        case ActivationType::tanh:
            return Eigen::tanh(x.array());
        case ActivationType::relu:
            return x.array() * (x.array()>0.).cast<double>();
        case ActivationType::sigmoid:
            return 1./(1. + exp(-x.array()));
        case ActivationType::linear:
            return x;
        case ActivationType::retanh:
            return Eigen::tanh(x.array()) * (x.array()>0.).cast<double>();
    }
}

Matrix activate(const Matrix &x, ActivationType phi){
    switch (phi) {
        case ActivationType::tanh:
            return Eigen::tanh(x.array());
        case ActivationType::relu:
            return x.array() * (x.array()>0.).cast<double>();
        case ActivationType::sigmoid:
            return 1./(1. + exp(-x.array()));
        case ActivationType::linear:
            return x;
        case ActivationType::retanh:
            return Eigen::tanh(x.array()) * (x.array()>0.).cast<double>();
    }
}

Vector derivate(const Vector &a, ActivationType phi){
    switch (phi) {
        case ActivationType::tanh:
            return 1. - a.array() * a.array();
        case ActivationType::relu:
            return (a.array() > 0.).cast<double>();
        case ActivationType::sigmoid:
            return a.array() * (1. - a.array());
        case ActivationType::linear:
            return Vector::Ones(a.size());
        case ActivationType::retanh:
            return (1. - a.array() * a.array()) * (a.array()>0.).cast<double>();
    }
}

Matrix derivate(const Matrix &a, ActivationType phi){
    switch (phi) {
        case ActivationType::tanh:
            return 1. - a.array() * a.array();
        case ActivationType::relu:
            return (a.array() > 0.).cast<double>();
        case ActivationType::sigmoid:
            return a.array() * (1. - a.array());
        case ActivationType::linear:
            return Vector::Ones(a.size());
        case ActivationType::retanh:
            return (1. - a.array() * a.array()) * (a.array()>0.).cast<double>();
    }
}

double inverse(double r, ActivationType phi){
    switch (phi) {
        case ActivationType::tanh:
            return atanh(r);
        case ActivationType::relu:
            return r; // note that relu is not strictly invertible; use wisely
        case ActivationType::sigmoid:
            return log(r/(1. - r));
        case ActivationType::linear:
            return r;
        case ActivationType::retanh:
            return atanh(r); // note that retanh is not strictly invertible; use wisely
    }
}
