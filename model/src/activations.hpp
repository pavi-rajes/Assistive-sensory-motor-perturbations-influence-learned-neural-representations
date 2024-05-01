/**
 * activations.hpp | bmi_model
 *
 * Activation functions and their derivatives.
 */
#ifndef activations_hpp
#define activations_hpp


#include "globals.hpp"

enum class ActivationType {
    tanh,
    relu,
    sigmoid,
    linear,
    retanh,
};

/**
 * Activation function.
 *
 * @param x     Preactivation or input
 * @param phi   Name of activation function (see ActivationType above)
 * @return      Activity (activation function applied to `x` elementwise)
 */
Vector activate(const Vector &x, ActivationType phi);
Matrix activate(const Matrix &x, ActivationType phi);

/**
 * Derivative of activation function.
 *
 * @param a     Activity (NOT the preactivation)
 * @param phi   Name of activation function (see ActivationType above)
 * @return      Elementwise derivative
 */
Vector derivate(const Vector &a, ActivationType phi);
Matrix derivate(const Matrix &a, ActivationType phi);

/**
 * Inverse of activation function, when possible.
 *
 * @param r     Activity (type: double)
 * @param phi   Name of activation function (see ActivationType above)
 * @return      Inverse of the activation function
 *
 * Examples:
 * 1) inverse(r, tanh) = atanh(r)
 * 2) inverse(r, relu) = r
 * In 2), relu is not invertible on all its domain.
 *
 * IMPORTANT NOTE: No safeguards were implemented to make sure that the inverse is meaningful for all values of r. Use wisely.
 */
double inverse(double r, ActivationType phi);


#endif /* activations_hpp */
