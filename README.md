# MLSNet

MLSNet: a deep learning model for predicting transcription factor binding sites.


# Dependencies

MLSNet works under Python 3.8

The required dependencies for MLSNet are as follows：

python==3.8.18

pytorch==2.1.1

numpy==1.24.1

pandas==1.4.4

scikit-learn==1.3.0

# Input

MLSNet takes two files as input: the Sequence file and the Shape file. The Sequence file is composed of two CSV files: one for training validation and one for testing. The datasets are available at http://cnn.csail.mit.edu/motif discovery/.The Shape file is computed from the corresponding DNA sequences in the Sequence file by the DNAshapeR tool, which can be downloaded from http://www.bioconductor.org/. The Shape file consists of ten CSV files of helix twist (HelT), minor groove width (MGW), propeller twist (ProT), rolling (Roll), and minor groove electrostatic potential (EP) for training validation data and testing data.

# Output

The "data" dataset is shown in the code as a sample. The train.py file is run to train the data and obtain the results. The model  is stored in models.
