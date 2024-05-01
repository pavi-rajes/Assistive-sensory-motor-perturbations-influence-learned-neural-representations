/**
 * VelocityKalmanFilter.hpp | bmi_model
 *
 * Description:
 * -----------
 * Velocity Kalman filter.
 *
 * Mathematical formulation:
 *
 * v[t] = Av[t-1] + w[t] (state evolution)
 *
 * z[t] = Hv[t] + q[t] (observation model)
 * where
 *     w[t] ~ Normal(0, W)
 *
 *     q[t] ~ Normal(0, Q)
 *
 *     v[t] = cursor velocity (in BCI context); typically [v_x, v_y]^T, with `dim = 2`
 *
 *     z[t] = neural activity
 *
 * Warning: Slightly different from the Kalman filter in Orsborn...Carmena, IEEE TNSRE, (2012), which uses
 * z[t] = Hx[t] + q[t] for the observation model, with x concatenating the position and velocity.
 *
 * Note: Member variable `m` implements the `1` concatenated to the state vector.
 */

#ifndef VelocityKalmanFilter_hpp
#define VelocityKalmanFilter_hpp

#include <vector>
#include "Effector.hpp"
#include "Readout.hpp"
#include "globals.hpp"
#include "Target.hpp"


enum class ReadoutSelectionMethod{
    random,         // select the first `nb_readout` units in the network with sufficient activity
    masking,        // mask a certain fraction of units and, among the remaining units, select `nb_readout` good units
    best_of_all,    // select the best `nb_readout` units
};


class VelocityKalmanFilter: public Effector {
    Matrix A, W, H, Q;                          // model parameters (see above)
    Vector m;                                   // mean activity 
    
    Vector v_hat;                               // velocity estimate
    Matrix P;                                   // error covariance
    
    bool is_trained;                            // whether the filer has been fitted once
    bool check_valid{ true };                   // whether to perform a selection of the readout units
    
    Readout readout;                            // 'measurement' object that read-out activity from the network
    std::vector<size_t> readout_unit_ids;       // IDs of the readout units
    std::vector<size_t> swapped_out_units;      // IDs of the readout units discarded in the unit swap, when swapping is used
    
    
public:
    VelocityKalmanFilter(Readout& _readout, int dim=2, double radius_for_reset=0.);
    void reset();
    void next_state(const Vector& network_activity);

    /**
     * Obtain matrices A, W, H and Q from data `kinematic_recordings` and `unit_recordings`.
     *
     * @param velocities recordings of the end_effector kinematic variables (shape = dim x time steps)
     * @param unit_recordings neural recordings (shape = nb_of_bci_units x time steps)
     */
    void encode(const Matrix& velocities, const Matrix& unit_recordings, const int trial_duration=50, const int burnin=0, ReadoutSelectionMethod selection_method=ReadoutSelectionMethod::best_of_all);
    
    
    /**
     * Decode the activity from `unit_recordings`.
     *
     * @param unit_recordings neural recordings (shape = nb_of_bci_units x time steps; notice that its dimensions are transposed to that of the encode function)
     * @todo normalize the shape of unit_recordings between decode and encode functions.
     */
    Matrix decode(const Matrix& unit_recordings);
    
    /// Compute Kalman gain.
    Matrix gain(const Matrix& P_minus);
    
    /// Get position and velocity of cursor
    Matrix get_end_effector_trajectory(const Matrix& unit_recordings);
    
    /**
     Perform closed-loop decoder adaptation, by updating matrices H and Q.
     */
    void clda(Matrix& velocities, const Matrix& unit_recordings, const std::vector<Vector> &targets, const int duration, const double alpha, const double beta);
    void rotate_velocities(Matrix& velocities, const std::vector<Vector> &targets, const int duration);
    void clda(Matrix& velocities, Matrix& unit_recordings, Target& targets, const int duration, const int hold_duration, const double alpha, const double beta);
    void rotate_velocities(Matrix& velocities, Target& targets, const int duration, const int hold_duration);
    
    ///Perform unit swapping
    void select_units_to_discard(const Matrix& bmi_unit_recordings, const int nb_units);
    void swap_units(Matrix& velocities, const Matrix& all_unit_recordings, const int trial_duration, const int nb_units, const int burnin=0);
    
    ///Turn off performing pre-selection of readout units (do that when using non-relu activation function)
    void do_not_preselect_readout_units() { check_valid = false; };
    
    ///Remove unit(s) from the readout set
    void remove_unit(const size_t id);
    void remove_units(const std::vector<size_t>& ids);
    
    ///Only keep units `ids` in the filter
    void only_include_units(const std::vector<size_t>& ids);
    
    ///Saving Kalman filter
    void save(std::string file_prefix);
    void load(std::string file_prefix);
    
    Effector* get_copy(){
        return new VelocityKalmanFilter(*this);
    };
    
    ///Getters
    Matrix get_A() {return A;}
    Matrix get_W() {return W;}
    Matrix get_H() {return H;}
    Matrix get_Q() {return Q;}
    Vector get_m() {return m;}
    std::vector<size_t> get_readout_unit_ids(){return readout_unit_ids;};
    Readout get_readout() {return readout;};
    int get_nb_channels() { return readout.get_nb_channels();}

};

#endif /* VelocityKalmanFilter_hpp */
