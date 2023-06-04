```bash
# install
source install.sh

# measure
python3 measure_ov_sd_pipe_latency.py

# for running these models first time on a new system requires model download
# set DRYRUN_BOOL=True in measure_ov_sd_pipe_latency.py 
# to download the model and dump the generated output for sanity check

# change DEVICE to cpu or gpu in the script

# odd situation in a770 env
# 1) openvino.runtime.core is required to be at the very beginning of, otherwise gpu device is not found by opencl. we overcome this by importing and calling check_ov_devices which import core.
# 2) even with the above, GPU is not found during debug mode in vscode. Further troubleshooting is needed here.
# 3) memory are not freed up after pipe is not used. Workaround: del pipe. "reproduce error by compiling many models on gpu"
```