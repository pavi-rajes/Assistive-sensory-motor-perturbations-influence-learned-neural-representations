CC = clang++

CFLAGS= -O3 -fopenmp -std=c++17 -march=native -stdlib=libc++ -I/usr/local/opt/libomp/include  -I../src  -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3

LDFLAGS=-L/usr/local/opt/libomp/lib



OBJECTS=ManifoldVelocityKalmanFilter.o FactorAnalysis.o PassiveCursor.o Target.o Gradient.o EligibilityTrace.o Input.o VelocityKalmanFilter.o Monitor.o PointMassArm.o rand_mat.o Readout.o TorqueBasedArm.o TwoLayerFFN.o globals.o utilities.o  DataGenerator.o OptimalLinearEstimator.o activations.o FFN.o RNN.o

SIMULATIONS=sim_bci_model sim_arm_model


.SECONDARY:

all: $(SIMULATIONS) 

sim_%: sim_%.o $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $< $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

%.o : ../src/%.cpp
	$(CC) $(CFLAGS) -c $<


clean: 
	rm -f *.o $(SIMULATIONS)
