/**
 * PassiveCursor.hpp | bmi_model
 *
 * Description:
 * -----------
 * Passively moving cursor, with constant velocity directed towards each target.
 *
 * Note:
 * ----
 * Rarely used.
 */
#ifndef PassiveCursor_hpp
#define PassiveCursor_hpp

#include "Effector.hpp"
#include "globals.hpp"

class PassiveCursor : public Effector{
    int duration;
    double distance_to_target;

public:
    /**
     * Constructor.
     *
     * Parameters:
     *  - duration: number of time step for reach
     *  - distance_to_target: in meters
     *  - dim: number of dimensions of workspace
     */
    PassiveCursor(int duration, double distance_to_target, int dim=2);
    
    Matrix get_end_effector_trajectory(const Matrix & unit_target_directions);
    
    void next_state(const Vector& unit_target_direction);
    Effector* get_copy(){return new PassiveCursor(*this);};

    void reset();
    int get_nb_channels(){ return 0; }


};


#endif /* PassiveCursor_hpp */
