/**
 * rnn4bci.hpp | bmi_model
 *
 * Description:
 * -----------
 * File typically included in the sim files.
 */
#ifndef rnn4bci_h
#define rnn4bci_h

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "activations.hpp"
#include "globals.hpp"
#include <random>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include "rand_mat.hpp"
#include "PointMassArm.hpp"
#include "TorqueBasedArm.hpp"
#include "Effector.hpp"
#include "Objectives.hpp"
//#include "RNN.hpp"  // already included in utilities.hpp
#include "Monitor.hpp"
#include "VelocityKalmanFilter.hpp"
#include "OptimalLinearEstimator.hpp"
#include "Readout.hpp"
#include "utilities.hpp"
#include "Input.hpp"
#include "EligibilityTrace.hpp"
#include "Target.hpp"
#include "DataGenerator.hpp"
#include "FFN.hpp"
#include "TwoLayerFFN.hpp"
#include "PassiveCursor.hpp"
#include "FactorAnalysis.hpp"
#include "ManifoldVelocityKalmanFilter.hpp"

#endif /* rnn4bci_h */
