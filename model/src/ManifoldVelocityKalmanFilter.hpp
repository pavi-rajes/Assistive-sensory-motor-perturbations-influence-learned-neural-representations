/**
 * ManifoldVelocityKalmanFilter.hpp | bmi_model
 *
 * Description:
 * -----------
 * This is velocity Kalman filter that takes into account the covariance of network activity, as per Sadtler et al., Nature (2014).
 * In a nutshell, instead of using the readout activity itself, we first find perform factor analysis to
 * extract the leading factors and then use the projections onto these factors to drive the Kalman filter.
 *
 * tl;dr
 * Same as VelocityKalmanFilter, except that factor analysis is perform on the readout activity and the steady-state Kalman gain is used.
 */
#ifndef ManifoldVelocityKalmanFilter_hpp
#define ManifoldVelocityKalmanFilter_hpp

#include <vector>
#include "Effector.hpp"
#include "Readout.hpp"
#include "globals.hpp"
#include "Target.hpp"
#include "FactorAnalysis.hpp"
#include "Objectives.hpp"

class ManifoldVelocityKalmanFilter: public Effector {
    Matrix A, W, H, Q;
    Matrix K_ss;    // steady-state Kalman gain
    Vector m_f;     // mean factor
    Vector s_f;     // std factor
    Vector m;       // mean activity
    Vector s;       // std activity
    
    Vector v_hat;
    Matrix P;
    
    bool is_trained;
    
    Readout readout;
    FactorAnalysis factor_analyser;
    
    std::vector<size_t> readout_unit_ids;
    
    
public:
    ManifoldVelocityKalmanFilter(Readout & _readout, FactorAnalysis & fa, int dim=2, double radius_for_reset=0.);

    /// Reset the effector.
    void reset();
    
    void next_state(const Vector& network_activity);

    /**
     * Obtain matrices A, W, H and Q from data `kinematic_recordings` and `unit_recordings`.
     *
     * @param velocities recordings of the end_effector kinematic variables (shape = dim x time steps)
     * @param unit_recordings neural recordings (shape = nb_of_bci_units x time steps)
     * @param trial_duration total duration of a trial (hold + move)
     * @param is_cursor_passive whether the effector is a PassiveCursor
     */
    void encode(const Matrix& velocities, const Matrix& unit_recordings, const int trial_duration=50, const bool is_cursor_passive=false);
    
    
    /**
     * Decode the activity from `unit_recordings`.
     *
     * @param unit_recordings neural recordings (shape = nb_of_bci_units x time steps)
     * @todo normalize the shape of unit_recordings between decode and encode functions.
     */
    Matrix decode(const Matrix& unit_recordings);
    
    /// Compute Kalman gain.
    Matrix gain(const Matrix& P_minus);
    
    /// Get position and velocity of cursor
    Matrix get_end_effector_trajectory(const Matrix & unit_recordings);
    
    /**
     Perform closed-loop decoder adaptation, by updating matrices H and Q.
     */
    void clda(Matrix& velocities, const Matrix& unit_recordings, const std::vector<Vector> &targets, const int duration, const double alpha, const double beta);
    void rotate_velocities(Matrix& velocities, const std::vector<Vector> &targets, const int duration);
    void clda(Matrix& velocities, Matrix& unit_recordings, Target& targets, const int duration, const int hold_duration, const double alpha, const double beta);
    void rotate_velocities(Matrix& velocities, Target& targets, const int duration, const int hold_duration);
    
    /**
     * Compute steady-state Kalman gain
     */
    Matrix steady_state_Kalman_gain();
    
    /**
     * Saving Kalman filter
     */
    void save(std::string file_prefix);
    //void load(std::string file_prefix);
    
    /**
     * Within-manifold perturbation.
     *
     * Replaces beta in the factor analysis model by P*beta, where P is a permutation matrix of the k-th order (k = nb of factors).
     */
    void within_manifold_perturbation(const Matrix & unit_recordings, Loss * loss, Target & target, const double target_loss, const int trial_duration, const int hold_duration=0, const double scale_factor=1., const double tolerance=5.);
    
    /**
     * Outside-manifold perturbation.
     *
     * Replaces beta in the factor analysis model by beta*P, where P is a permutation matrix of the p-th order (p = nb of readout units).
     */
    void outside_manifold_perturbation(const Matrix & unit_recordings, Loss * loss, Target & target, const double target_loss, const int trial_duration, const int hold_duration=0, const double scale_factor=1., const double tolerance=5.);
    
    
    Vector compute_openloop_velocity(const Matrix& network_activity);
    
    static Vector compute_angles(const Matrix& v1, const Matrix& v2);
    
    std::vector<size_t> get_readout_unit_ids(){return readout_unit_ids;};

    Effector* get_copy(){return new ManifoldVelocityKalmanFilter(*this);};

    /**
     Getters
     */
    Matrix get_A() {return A;}
    Matrix get_W() {return W;}
    Matrix get_H() {return H;}
    Matrix get_Q() {return Q;}
    Vector get_m() {return m;}
    Readout get_readout(){return readout;}

};




#endif /* ManifoldVelocityKalmanFilter_hpp */
