/**
 * Effector.hpp | bmi_model
 *
 * An effector is either an arm (e.g. PointMassArm) or a decoder (e.g. VelocityKalmanFilter).
 * This class is not instantiated by itself (see virtual member function).
 */
#ifndef Effector_hpp
#define Effector_hpp

#include "globals.hpp"
#include <iostream>
#include <random>

enum EffectorType {NONE, ARM, DECODER, PASSIVECURSOR};  //TODO: replace by scoped enum
enum ArmType {None, PointMass, TorqueBased}; //TODO: replace by scoped enum


class Effector {

protected:
    int _dim;   // number of dimensions of movement (x,y) => dim=2; (x,y,z) => dim=3
    std::mt19937 rng_for_reset;
    double radius_for_reset;

public:
    EffectorType effector_type;
    ArmType arm_type;
    
    Vector initial_state, end_effector_init_cond;  // initial conditions for the effector
    
    Vector state; // (position, velocity, force) for PointMassArm, (angles, angular velocities, torques) for TorqueBasedArm
    
    Vector end_effector; // position, velocity of end-effector and force applied
    
    
    Effector(int dim, double radius_for_reset=0.) : _dim(dim), effector_type(NONE), arm_type(None), radius_for_reset(radius_for_reset) {
        initial_state = Vector::Zero(3*_dim);
        end_effector_init_cond = Vector::Zero(3*_dim);
        end_effector = Vector::Zero(3*_dim);
        state = Vector::Zero(3*_dim);
    }
    
    virtual Matrix get_end_effector_trajectory(const Matrix & ctrls) {
        return ctrls;
    }
    
    int get_dim() {return _dim;}
    
    std::string get_effector_type() {
        std::string s;
        switch (effector_type) {
            case NONE:
                s = "no effector";
                break;
            case ARM:
                s = "arm";
                break;
            case DECODER:
                s = "decoder";
                break;
            case PASSIVECURSOR:
                s = "passive cursor";
                break;
            default:
                break;
        }
        return s;
    }
    
    virtual void next_state(const Vector& controls) {}
    
    /// Reset the effector.
    virtual void reset() {
        state = initial_state;
        end_effector = end_effector_init_cond;
    }
    
    virtual ~Effector() {};
    virtual Effector* get_copy(){
        return new Effector(*this);
    };

    
    /// To remove unit (for unit swap or dropout) in the decoder; never used with the arm
    /// TODO: implement a decoder class inheriting from effector and from which all decoders will inherit.
    void remove_unit(size_t id){};
    virtual int get_nb_channels() { return 0;}

};

#endif /* Effector_hpp */
