/**
 * Objectives.hpp | bmi_model
 *
 * Description:
 * -----------
 * Repository of training objectives.
 *
 * Notes:
 * -----
 * 1. In defining any new training objectives, the declaration structure
 * of the overloaded ( ) operator should be observed.
 * 2. This is header only, for some reason. I don't remember why (perhaps I wanted a template class...).
 *
 * Todo:
 * ----
 * Create .cpp so that Makefile recompile this correctly.
 */
#ifndef Objectives_hpp
#define Objectives_hpp
#include <Eigen/Dense>
#include <iostream>
#include "globals.hpp"

///Abstract loss class
struct Loss {
    double effort_penalty;
    double activity_regularizer;
    
    Loss(double e_p, double a_r) : effort_penalty(e_p), activity_regularizer(a_r) {}
    
    virtual double operator() (const Matrix& states, const Vector& target, int T_prep=0, double scale_factor=1.) =0;
    virtual double operator() (const Vector& state, const Vector& target, int time_until_end, double scale_factor=1.) =0;

    virtual ~Loss() {};
};

///L2 penalty on final position of the end_effector/cursor, as well as a L2 penalties on zero final velocity and force.
struct EndLoss : public Loss {
    
    double hyperparam_v;
    double hyperparam_f;
    
    EndLoss(double h_v, double h_f, double e_p=0., double a_r=0.) : hyperparam_v(h_v), hyperparam_f(h_f), Loss(e_p, a_r) {}
    
    double operator() (const Matrix& states, const Vector& target, int T_prep=0, double scale_factor=1.) {
        int duration = int(states.cols());
        double total_loss{ 0. };
        Vector center_target = Vector::Zero(target.size());
        
        // Hold interval
        for (int t=-T_prep;t<0;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, center_target, -1-t, scale_factor);
        }
        // Move interval
        for (int t=0;t<duration-T_prep;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, target, duration-T_prep-1-t, 1.);
        }
        
        return total_loss;
    }
    
    double operator() (const Vector& state, const Vector& target, int time_until_end, double scale_factor=1.) {
        
        if (time_until_end > 0) return 0;
        else {
            double position_cost(0.);
            for (long i(0);i<target.size();i++) position_cost += (state(i) - target(i)) * (state(i) - target(i)) / (0.01*0.01);  // DEBUG!!!;
            
            double velocity_cost(0.);
            for (long i=target.size(); i<2*target.size(); i++) {
                velocity_cost += state(i) * state(i) / (0.02*0.02); // DEBUG!!!
            }
            velocity_cost *= hyperparam_v;
            
            double force_cost(0.);
            for (long i=2*target.size(); i<3*target.size(); i++) {
                force_cost += state(i) * state(i) / (0.08*0.08); // DEBUG!!!
            }
            force_cost *= hyperparam_f;
            
            return scale_factor*(position_cost + velocity_cost + force_cost);
        }
    }
    
};


///Cumulative L2 loss computed as an average over the last `nb_steps` of the movement.
struct CumulativeEndLoss : public Loss {
    
    int nb_steps;
    double target_radius;
    double distance_to_target;
    
    CumulativeEndLoss(double nb_steps, double effort_penalty, double target_radius, double distance_to_target, double a_r=0.) : nb_steps(nb_steps), Loss(effort_penalty, a_r), target_radius(target_radius), distance_to_target(distance_to_target) {}
    

    double operator() (const Matrix& states, const Vector& target, int T_prep=0, double scale_factor=1.) {
        int duration = int(states.cols());
        double total_loss{ 0. };
        Vector center_target = Vector::Zero(target.size());
        
        // Hold interval
        for (int t=-T_prep;t<0;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, center_target, -1-t, scale_factor);
        }
        // Move interval
        for (int t=0;t<duration-T_prep;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, target, duration-T_prep-1-t, 1.);
        }
        
        return total_loss;
    }
    
    
    double operator() (const Vector& state, const Vector& target, int time_until_end, double scale_factor=1.) {
        if (time_until_end > nb_steps) return 0.;
        else {            
            double criterion{ 0. };
            //if (target.norm() > 1e-8) {
                criterion = target_radius;
            //}
            Vector position(target.size());
            for(int i(0);i<position.size();i++) position(i) = state(i);
            
            double loss(0);
            double imp_to_target_distance = (position - target).norm();

            if (imp_to_target_distance > criterion) {
                loss += scale_factor * (imp_to_target_distance - criterion) * (imp_to_target_distance - criterion) / (0.01*0.01);
            }
            else loss = 0.;
            return loss / (1.*nb_steps);
        }
    }
};

///Cumulative L2 loss on position and velocity computed as an average over the last `nb_steps` of the movement.
struct CumulativeEndLossWithVelocity : public Loss {
    
    int nb_steps;
    double target_radius;
    double distance_to_target;
    double hyperparam_v;
    double hyperparam_f;

    
    CumulativeEndLossWithVelocity(double nb_steps, double effort_penalty, double h_v, double h_f, double target_radius, double distance_to_target, double a_r=0.) : nb_steps(nb_steps), Loss(effort_penalty, a_r), hyperparam_v(h_v), hyperparam_f(h_f), target_radius(target_radius), distance_to_target(distance_to_target) {}
    
    double operator() (const Matrix& states, const Vector& target, int T_prep=0, double scale_factor=1.) {
        int duration = int(states.cols());
        double total_loss{ 0. };
        Vector center_target = Vector::Zero(target.size());
        
        // Hold interval
        for (int t=-T_prep;t<0;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, center_target, -1-t, scale_factor);
        }
        // Move interval
        for (int t=0;t<duration-T_prep;t++){
            Vector s = states.col(t);
            total_loss += this->operator()(s, target, duration-T_prep-1-t, 1.);
        }
        
        return total_loss;
    }
    
    
    double operator() (const Vector& state, const Vector& target, int time_until_end, double scale_factor=1.) {

        if (time_until_end > nb_steps) return 0.;
        else {
                const int dim{ int(target.size()) };
            
                double loss{ 0. };
                
                // "Force" cost
                for (int i=2*dim; i<3*dim; i++) loss += scale_factor * hyperparam_f * state(i) * state(i);
            
                // Velocity cost
                for (int i=dim; i<2*dim; i++) loss += scale_factor * hyperparam_v * state(i) * state(i);
                    
                // Position cost
                double criterion{ 0. };
                if (target.norm() > 1e-8) criterion = target_radius;
                Vector position(dim);
                for(int i(0);i<position.size();i++) position(i) = state(i);

                double effector_to_target_distance = (position - target).norm();

                if (effector_to_target_distance > criterion) {
                    loss += scale_factor * (effector_to_target_distance - criterion) * (effector_to_target_distance - criterion);
                } else {
                    loss += 0.;
                }
                return loss / (1.*nb_steps);
        }
    }
};

#endif /* Objectives_hpp */
