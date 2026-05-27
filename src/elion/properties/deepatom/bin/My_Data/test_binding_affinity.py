import subprocess
import os
import time
import numpy
import pytest

# Configuration
# SCRIPT = "/blue/lic/share/local/deepatom/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
SCRIPT = "/blue/lic/huangzihang/repos/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
SCRIPT_ARGS = ["-t", "vs", "-d", "/blue/lic/huangzihang/repos/deepatom/bin/My_Data"]
OUTPUT_FILE = "/blue/lic/huangzihang/repos/deepatom/bin/My_Data/vs_My_Data.csv"
CONDA_ENV = "/blue/lic/huangzihang/repos/miniconda3/envs/binding_affinity_27"
CONDA_SH = "/blue/lic/huangzihang/repos/miniconda3/etc/profile.d/conda.sh"
ACCEPTABLE_STDDEV = 0.5
NUM_RUNS = 10
MAX_WAIT_SECONDS = 10  # Maximum time to wait for file update

def test_binding_affinity_stddev():
    """Test that the standard deviation of binding affinity predictions is within acceptable range."""
    # Clear the output file
    open(OUTPUT_FILE, 'w').close()

    # Collect binding affinity values
    values = []
    for i in range(NUM_RUNS):
        print "Running iteration %d..." % (i + 1)
        # Run the script in the conda environment using bash -c
        cmd = '. %s && conda activate %s && bash %s %s' % (
            CONDA_SH, CONDA_ENV, SCRIPT, ' '.join(SCRIPT_ARGS)
        )
        try:
            # Use check_output to capture stdout and stderr, raise if non-zero exit
            output = subprocess.check_output(
                ['bash', '-c', cmd],
                stderr=subprocess.STDOUT  # Capture stderr with stdout
            )
            print "Iteration %d output: %s" % (i + 1, output)
        except subprocess.CalledProcessError as e:
            print "Iteration %d error: %s" % (i + 1, e.output)
            raise pytest.fail("Command failed: %s" % e.output)

        # Wait for the output file to be written
        start_time = time.time()
        while True:
            if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
                break
            if time.time() - start_time > MAX_WAIT_SECONDS:
                raise pytest.fail(
                    "Output file %s was not written after %d seconds in iteration %d" % 
                    (OUTPUT_FILE, MAX_WAIT_SECONDS, i + 1)
                )
            time.sleep(0.1)  # Poll every 100ms

        # Read the last line of the output file
        with open(OUTPUT_FILE, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise pytest.fail("Output file %s is empty after iteration %d" % (OUTPUT_FILE, i + 1))
            last_line = lines[-1].strip()
            try:
                value = float(last_line.split(',')[1])
                values.append(value)
            except (IndexError, ValueError) as e:
                raise pytest.fail("Invalid CSV format in %s: %s, error: %s" % (OUTPUT_FILE, last_line, e))

    # Calculate mean and standard deviation
    mean = numpy.mean(values)
    stddev = numpy.std(values, ddof=1)  # Use ddof=1 for sample standard deviation

    # Log results
    print "Outputs: %s" % values
    print "Mean: %s" % mean
    print "Standard Deviation: %s" % stddev

    # Assert standard deviation is within acceptable range
    assert stddev <= ACCEPTABLE_STDDEV, \
        "Standard deviation %s exceeds acceptable range (%s)" % (stddev, ACCEPTABLE_STDDEV)