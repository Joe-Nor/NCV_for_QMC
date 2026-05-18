#!/bin/bash
# Compile the twist statistics version

gfortran -O3 -o rsse_twist_stats rsse_update_loops_cursor_optimized_v3_twist_stats.f90

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Executable: rsse_twist_stats"
else
    echo "Compilation failed!"
    exit 1
fi
