TYPE=$2
MACHINE=$1

if [[ "$TYPE" == "mesh" ]]; then
    ../venv/bin/python3 generate/mesh.py $MACHINE
elif [[ "$TYPE" == "sim" ]]; then
    ../venv/bin/python3 generate/simulation.py --machine=$MACHINE --array_id 0 --array_total 1
elif [[ "$TYPE" == "conv" ]]; then
    ../venv/bin/python3 generate/conversion.py --machine=$MACHINE
else
    echo "Invalid type."
fi
