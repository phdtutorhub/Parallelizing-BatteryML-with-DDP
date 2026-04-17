# Parallelizing-BatteryML-with-DDP

You can donwload BatteryML by git clone https://github.com/microsoft/BatteryML.

This repo contains all 5 files needed to run a parallel version of BatteryML with DDP on HPC.
The 5 files are annotated with comments.  They are tested on Polaris at ALCF and ran successfully up to 400 nodes.  If your system does not support PBS, you need to modify the launch script baseline.sh accordingly.  You also need to modify lines that refer to the root directory of BatteryML and your conda directory.
The rest of the files need to be renamed by removing the .com suffix and be copied to the appropriate directories as follow:

baseline.sh -> BatteryML/baseline.sh

baseline.py.com -> BatteryML/baseline.py

pipeline.py.com -> BatteryML/batteryml/pipeline.py

nn_model.py.com -> BatteryML/batteryml/models/nn_model.py

MATR_split.py.com -> BatteryML/batteryml/train_test_split/MATR_split.py

Before you copy over those files in the corresponding directories, you should do the following for backup:

mv pipeline.py pipeline.py.bak

mv nn_model.py nn_model.py.bak

mv MATR_split.py MATR_split.py.bak

A companion paper bat-ddp.pdf is included.

Please reference this site if you are using the codes and the paper.
