#!/bin/bash

NUM_SEEDS=1 # NUM_SEEDS : number of realizations
N_TARGETS=8 # number of targets
N_REALS=5  # number of noisy trajectory to simulate for a given seed
HOLD_DURATION=0
DATADIR="../data/"
PARAMDIR="../params/"
RESULTDIR="../results/"
DIRNAME="arm-model/"

PREDIR="${DATADIR}${DIRNAME}"

mkdir -p $PREDIR
mkdir -p "${PARAMDIR}${DIRNAME}"

cp sim_arm_model.cpp "${PARAMDIR}${DIRNAME}copy_sim_arm_model.cpp"
cp ../src/RNN.cpp "${PARAMDIR}${DIRNAME}copy_RNN.cpp"

make

for (( i=1; i <= $NUM_SEEDS; i++ ))
	do
		DIR="${PREDIR}seed$i/" 
		mkdir -p $DIR 
		./sim_arm_model --seed $i --dir $DIR --n_targets $N_TARGETS --n_reals $N_REALS --hold_duration $HOLD_DURATION
        cd ../analysis
        python plot_trajectories.py -n_targets $N_TARGETS -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "manual_before_learning"
        python plot_trajectories.py -n_targets $N_TARGETS -n_reals $N_REALS -hold_duration $HOLD_DURATION -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "naive"
	python plot_loss.py -resultdir "${RESULTDIR}${DIRNAME}" -datadir "${DATADIR}${DIRNAME}" -seed $i -type "manual_before_learning" -subsampling 1
        cd ../sim
	done

