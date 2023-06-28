#!/bin/bash
echo $1

time /opt/SimulationsPlus/bin/RunAP.sh -t SMI $1 -m TOX,GLB -N 16
