#!/bin/bash

N_SEEDS=1 # N_SEEDS : number of seeds
N_TARGETS=8
N_TARGETS_GEN=16
N_REALS=5
HOLD_DURATION=0
DATADIR="../data/"
PARAMDIR="../params/"
RESULTDIR="../results/"

# Compile 
make

for CLDA in 1 0.9 0.75 0.5
	do
		DIRNAME="bci-model-clda${CLDA}/"
		PREDIR="${DATADIR}${DIRNAME}"

		# Force-create the data directory
		mkdir -p $PREDIR

		# Force-create the param directory 
		mkdir -p "${PARAMDIR}${DIRNAME}"

		# Copy simulation file and the RNN and FFN infos to param directory
		cp sim_bci_model.cpp "${PARAMDIR}${DIRNAME}copy_sim_bci_model.cpp"
		cp ../src/RNN.cpp "${PARAMDIR}${DIRNAME}copy_RNN.cpp"
		cp ../src/FFN.cpp "${PARAMDIR}${DIRNAME}copy_FNN.cpp"

		# Loop over random initialization of the model
		for (( i=1; i <= $N_SEEDS; i++ ))
			do
				DIR="${PREDIR}seed$i/" 
				mkdir -p $DIR 
				./sim_bci_model --seed $i --dir $DIR --clda $CLDA --n_targets $N_TARGETS --n_reals $N_REALS --hold_duration $HOLD_DURATION
				cd ../analysis
				python plot_trajectories.py -n_targets $N_TARGETS -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "bci"
			 	python plot_trajectories.py -n_targets $N_TARGETS -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "bci_before_learning"
				python plot_trajectories.py -n_targets $N_TARGETS -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "manual_with_bci_context"
                python plot_trajectories.py -n_targets $N_TARGETS_GEN -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "gen_bci"
				python plot_loss.py -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "bci" -subsampling 100
				cd ../sim
			done
	done
