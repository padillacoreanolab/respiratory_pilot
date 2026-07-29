# respiratory_pilot
Read in TDT files analyze resp data. 
# notes for furture self
created repository with gitignore file and read me file 
gitignore is a file used to make sure data is not deposited to github we can add different file types to be ignored by adding the file type to the ignore list e.g. *.mp4 using astrix 
astrix are place holders that anything before is ok but anything after will be ignored
readme will explain the notebook 
resp env was created with 
python required for spikeinterface is 3.10.9
numpy verison is rewuired is 1.26.4

Environment Setup

A Conda environment file (`environment.yml`) is included for reproducibility.

### Create the environment
```
conda env create -f environment.yml
conda activate biopipeline-env
```
### Keypoint-MoSeq
```
Create a new conda environment with python 3.10:

conda create -n keypoint_moseq python=3.10
conda activate keypoint_moseq
Then use pip to install the version of keypoint moseq that you want:

pip install keypoint-moseq # CPU only
pip install keypoint-moseq[cuda] # GPU with CUDA 12
To run keypoint-moseq in jupyter, either launch jupyterlab directly from the keypoint_moseq environment or register a globally-accessible jupyter kernel as follows:

python -m ipykernel install --user --name=keypoint_moseq
```
ctrl+shift+l then backspace this is a cell control funtion that allows me to edit my cell info by being able to erase the same words in a line and then type in those line the word i want to replace 
