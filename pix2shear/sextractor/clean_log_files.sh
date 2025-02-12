#!/bin/bash

logs_dir="/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs"
if [[ ! -d "$logs_dir" ]]; then
    echo "ERROR: Logs directory does not exist: $logs_dir"
    exit 1
fi

rm -f "$logs_dir"/*.log

if [[ $? -eq 0 ]]; then
    echo "Successfully deleted all log files in $logs_dir."
else
    echo "ERROR: Failed to delete log files!"
    exit 1
fi
