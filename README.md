# Molecules Properties Prediction

This project is dedicated to molecular property prediction using Machine Learning Models, such as Random Forest, Graph Isomorphism Networks (GIN).

## Installation

### Prerequisites

Make sure you have Python and pip installed on your system. If not, you can download and install them from [python.org](https://www.python.org/).

### Install Pipenv

The project uses Pipenv for dependency management. If you don't have Pipenv yet, you can install it using the following command:

```bash
pip install pipenv
```
Set up the Environment

Clone the GitHub repository to your local machine:
```
git clone https://github.com/emanieluu/aml_project.git
```

Navigate to the project directory:
```
cd aml_project
```

Install dependencies with Pipenv:
```
pipenv install
```

To train a model, run the following command:
```
python -m molprop_prediction.scripts.train
```
This command will execute the model training script. Make sure you have the necessary data available and properly configured in the script.

## Project Structure

molprop_prediction/: The main package of the project.  
models/: Contains model definitions.  
scripts/: Contains Python scripts for training, prediction, etc.  
configs/: Configuration files for model hyperparameters.  
data/: Folder for storing data required for training and evaluation.  
docs/: Project documentation.  
tests/: Unit and integration tests.

## Contributors

Claire Dechaux, François Lacarce, Emanie Luu 
