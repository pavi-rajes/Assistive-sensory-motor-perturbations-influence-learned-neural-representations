/**
 * OptimalLinearEstimator.hpp | bmi_model
 *
 * Description:
 * -----------
 * Optimal linear estimator of velocity, as per Salinas & Abbott (1994). See also Koyama...Kass J Comput Neurosci (2010)
 *
 * The activity of unit i is assumed to be given by
 *              y_i = bias_i + b_i_x*vx + b_i_y*vy + noise
 * We further define
 *      m_i*p_i = b_i
 * with m_i = modulation depth (scalar) = | b_i |
 *      p_i = preferred direction (unit vector)
 *      b_i = (b_i_x,b_i_y)^T
 *
 * Encoding:
 * The values of beta_i = (bias_i, b_i_x, b_i_y) are obtained by simple linear regression (assuming homoscedastic noise):
 *      Y_i = X_i @ beta_i
 * where
 *      Y_i : vector of activity (one element for each time point)
 *      X_i : vertical concatenation of vectors of the form (1, v_x, v_y)
 *
 * Decoding:
 * Observing z_t = (y_t - bias)./m = Pv_t
 * where y_t, bias, m are size = readout size.
 * Solve (d/dv_t) (1/2)|z_t - Pv_t|^2 = 0 for v_t.
 * => v_t = (P^TP)^{-1}P^T z_t
 */
#ifndef OptimalLinearEstimator_hpp
#define OptimalLinearEstimator_hpp

#include "Effector.hpp"
#include "Readout.hpp"
#include "globals.hpp"
#include <tuple>

enum class OLEtype{
    velocity,
    target
};

class OptimalLinearEstimator: public Effector  {
    Readout readout;
    Vector b;   // biases
    Vector m;   // modulation
    Matrix P;   // preferred directions (nb readout units x dim)
    OLEtype encode_type;
    bool is_randomly_initialized;
    double conversion_factor; // m/s
    bool is_trained;
    void _encode_velocities(const Matrix& velocities, const Matrix& network_activity);
    void _encode_targets(const Matrix& targets, const Matrix& network_activity, const int duration, const int hold_duration);
    double threshold_for_readout;
    std::vector<size_t> readout_unit_ids;
    
public:

    OptimalLinearEstimator(Readout & _readout, int dim=2, OLEtype encoding_type=OLEtype::target, double radius_for_reset=0.);
    void reset();
    void init_parameters();
    void encode(const Matrix& velocities_or_targets, const Matrix& unit_recordings, const int duration, const int hold_duration);
    void next_state(const Vector& network_activity);
    Matrix get_end_effector_trajectory(const Matrix & unit_recordings);
    Readout get_readout(){return readout;}
    
    /**
     * Rotate by an angle a fraction f of readout units
     *
     * Parameters:
     *  - angle: angle in degrees
     *  - f: fraction of readout units whose preferred directions will be rotated
     */
    void rotate_pds(double angle_in_degrees, double f);
    std::vector<size_t> get_readout_unit_ids(){return readout_unit_ids;};
    Effector* get_copy(){return new OptimalLinearEstimator(*this);};

};


#endif /* OptimalLinearEstimator_hpp */
