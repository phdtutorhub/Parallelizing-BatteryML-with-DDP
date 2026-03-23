#!/bin/bash -l
#PBS -l select=<number-of-nodes>:system=<your-system>
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:<your-file-system-of-choice>
#PBS -q <your-queue-of-choice>
#PBS -A <your-allocation>

# MPI example w/ <PPN> MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=<PPN>

echo "=================================="
echo "Starting MPI job at $(date)"
echo "=================================="
echo ""

source <your-conda-directory>/bin/activate
cd <your-BatteryML-directory>
cat $PBS_NODEFILE > <your-BatteryML-directory>/nodelist
MASTER_ADDR=$(head -n 1 <your-BatteryML-directory>/nodelist)
MASTER_PORT=29500
HOSTLIST=$(cat <your-BatteryML-directory>/nodelist | uniq)

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE=${NRANKS_PER_NODE}"

RANK=0
for node in $HOSTLIST; do
  echo "Launching on node $node with rank $RANK"
  ssh $node "source <your-conda-directory>/bin/activate && \
	cd <your-BatteryML-directory> && \
	  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NRANKS_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/atang/BatteryML/baseline.py" &
  RANK=$((RANK+1))
done

wait

echo ""
echo "====================================="
echo "MPI job finished at $(date)"
echo "====================================="

