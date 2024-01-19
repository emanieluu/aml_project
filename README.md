# Molecules Properties Prediction

This project is dedicated to molecular property prediction using Machine Learning Models, such as Random Forest, Graph Isomorphism Networks (GIN) and Graph Attention Transformers (GAT).

## Installation

### Prerequisites

Make sure you have Python and pip installed on your system. If not, you can download and install them from [python.org](https://www.python.org/).

### Development Setup

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

### Data Preprocessing

We perform two types of preprocessing for the data initially provided in SMILES format:

1. **Tabular Data Processing:**
   - The data is preprocessed into tabular format with feature engineering, resulting in 100 features.
   - The feature engineering process is time-consuming, and for convenience, we store the tabular data in two folders: `tabular_feature_enhancement` and `tabular_preprocessed_data`. This allows us to directly access these preprocessed data when training the models.
   - To initiate the preprocessing of these data, execute the following command:
     ```
     python -m molprop_prediction.scripts.preprocess_tabular
     ```

2. **Graph Data Processing:**
   - The data is preprocessed into a format suitable for graph neural networks. This process is quick and is performed just before model training when executing the training file with GIN or GAT input.


### Model Training 

To train a model, run the following command:
```
python -m molprop_prediction.scripts.train
```
This command will execute the model training script. Follow the prompt instructions. Make sure you have the necessary data available and properly configured in the script.

### Hyperparameter Optimisation 

To train a model, run the following command:
```
python -m molprop_prediction.scripts.grid_search
```

## Folder structure


.
└── aml_project/
    ├── data/
   │   ├── predictions/          #contains the results 
    │   │   ├── GAT
    │   │   ├── GIN
    │   │   └── RF
    │   ├── raw_data
    │   ├── tabular_feature_enhancement
    │   └── tabular_preprocessed_data
    ├── molprop_prediction /
    │   ├── configs/
    │   │   ├── GAT
    │   │   ├── GIN
    │   │   └── RF
    │   ├── grid_results
    │   ├── models
    │   ├── scripts
    │   └── trained_models
    ├── notebooks
    └── reports/
        └── figures 

## Contributors

Claire Dechaux, François Lacarce, Emanie Luu 
