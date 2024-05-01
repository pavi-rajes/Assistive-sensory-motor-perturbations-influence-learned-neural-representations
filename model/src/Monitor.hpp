/**
 * Monitor.hpp | bmi_model
 *
 * Description:
 * -----------
 * Class to record (log) snapshots or sequences of snapshots of vectorial simulation variables (e.g, network activity, kinematics, but not weight matrices).
 */

#ifndef Monitor_hpp
#define Monitor_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include "globals.hpp"

class Monitor {
    std::vector<Vector> buffer;
    
public:
    int nb_of_simultaneous_recordings();
    int get_buffer_size();
    
    ///Clear buffer content
    void reset();
    
    ///Record a single snapshop of the measured quantity
    void record_snapshot(const Vector & snap);
    
    ///Record an entire sequence of the measured quantity
    void record_sequence(const Matrix & seq);
    void record_sequence(const std::vector<Vector> & seq);
    
    ///Get data in buffer
    Matrix get_data();
    Matrix get_data(unsigned start, unsigned end);
    Matrix get_data(int last_n);

    ///Output to file
    void save(std::string file_name);
};


#endif /* Monitor_hpp */
