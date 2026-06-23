#!/bin/bash
set -euo pipefail

# Script to submit multiple sbatch jobs in serial dependency chain
# Usage: bash run_multi.sh <number_of_jobs> [sbatch_script] [initial_dependency_jobid]
# Example: bash run_multi.sh 5 xxx.sub
# Example with dependency: bash run_multi.sh 5 xxx.sub 12345

# Check if number of jobs is provided
if [ $# -lt 1 ]; then
    echo "Error: Please provide the number of jobs to submit"
    echo "Usage: bash run_multi.sh <number_of_jobs> [sbatch_script] [initial_dependency_jobid]"
    echo "Example: bash run_multi.sh 5 xxx.sub"
    echo "Example with dependency: bash run_multi.sh 5 xxx.sub 12345"
    exit 1
fi

NUM_JOBS=$1
SBATCH_SCRIPT=${2:-"xxx.sub"}  # Default to xxx.sub if not provided
INITIAL_DEPENDENCY=${3:-}  # Optional initial job dependency

# Check if the sbatch script exists
if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Error: Sbatch script '$SBATCH_SCRIPT' not found"
    exit 1
fi

# Validate number of jobs
if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || [ "$NUM_JOBS" -lt 1 ]; then
    echo "Error: Number of jobs must be a positive integer"
    exit 1
fi

echo "Submitting $NUM_JOBS jobs in serial dependency chain..."
echo "Using sbatch script: $SBATCH_SCRIPT"
if [ -n "$INITIAL_DEPENDENCY" ]; then
    echo "Initial dependency: Job ID $INITIAL_DEPENDENCY"
fi
echo "----------------------------------------"

PREV_JOBID="$INITIAL_DEPENDENCY"

for i in $(seq 1 $NUM_JOBS); do
    if [ -z "$PREV_JOBID" ]; then
        # First job - no dependency
        echo "Submitting job $i (no dependency)..."
        OUTPUT=$(sbatch "$SBATCH_SCRIPT")
        JOBID=$(echo "$OUTPUT" | awk '{print $NF}')
        echo "Job $i submitted with ID: $JOBID"
    else
        # Jobs with dependency (including first job if initial dependency is set)
        if [ $i -eq 1 ]; then
            echo "Submitting job $i (depends on initial job $PREV_JOBID)..."
        else
            echo "Submitting job $i (depends on job $PREV_JOBID)..."
        fi
        OUTPUT=$(sbatch --dependency=afterany:$PREV_JOBID "$SBATCH_SCRIPT")
        JOBID=$(echo "$OUTPUT" | awk '{print $NF}')
        echo "Job $i submitted with ID: $JOBID (dependency: afterany:$PREV_JOBID)"
    fi
    if ! [[ "$JOBID" =~ ^[0-9]+$ ]]; then
        echo "Error: Failed to parse Slurm job ID from sbatch output: $OUTPUT" >&2
        exit 1
    fi
    
    # Update previous job ID for next iteration
    PREV_JOBID=$JOBID
    echo ""
done

echo "----------------------------------------"
echo "All $NUM_JOBS jobs submitted successfully!"
echo "Job dependency chain created."
