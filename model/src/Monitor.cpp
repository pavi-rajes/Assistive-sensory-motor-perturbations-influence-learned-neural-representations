/**
 * Monitor.cpp | bmi_model
 */
#include "Monitor.hpp"
#include <fstream>
void Monitor::reset() {
    buffer.clear();
}

void Monitor::record_snapshot(const Vector & snap){
    buffer.push_back(snap);
}

void Monitor::record_sequence(const Matrix & seq){
    for (int c(0);c<seq.cols();c++){
        buffer.push_back(seq.col(c));
    }
}

void Monitor::record_sequence(const std::vector<Vector> & seq){
    for (int c(0);c<seq.size();c++){
        buffer.push_back(seq[c]);
    }
}

int Monitor::get_buffer_size(){
    return int(buffer.size());
}

int Monitor::nb_of_simultaneous_recordings(){
    int n(0);
    if (buffer.size()>0){
        n = int(buffer[0].size());
    }
    return n;
}

Matrix Monitor::get_data(){
    int T = get_buffer_size();
    Matrix D(nb_of_simultaneous_recordings(), T);
    for (int t(0);t<T;t++){
        D.col(t) = buffer[t];
    }
    return D;
}

Matrix Monitor::get_data(unsigned start, unsigned end){
    Matrix D(nb_of_simultaneous_recordings(), end-start);
    for (int t(start);t<end;t++){
        D.col(t-start) = buffer[t];
    }
    return D;
}

Matrix Monitor::get_data(int last_n){
    Matrix D(nb_of_simultaneous_recordings(), last_n);
    int c{ 0 };
    for (auto p=buffer.end()-last_n;p<buffer.end();p++){
        D.col(c) = *p;
        c++;
    }
    return D;
}

void Monitor::save(std::string file_name){
    std::ofstream f(file_name);
    int T = get_buffer_size();
    int N = nb_of_simultaneous_recordings();
    
    for (int t(0);t<T;t++){
        for(int i(0);i<N-1;i++){
            f <<buffer[t].data()[i]<<"\t";
        }
        f<<buffer[t].data()[N-1]<<std::endl;
    }
    f.close();
}
