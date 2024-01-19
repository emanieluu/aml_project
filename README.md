# Molecules Properties Prediction

This project is dedicated to molecular property prediction using Machine Learning Models, such as Random Forest, Graph Isomorphism Networks (GIN) and Graph Attention Transformers (GAT).

## 1 Installation

### 1.1 Prerequisites

Make sure you have Python and pip installed on your system. If not, you can download and install them from [python.org](https://www.python.org/).

### 1.2 Development Setup

The project uses Pipenv for dependency management. If you don't have Pipenv yet, you can install it using the following command:

```bash
pip install pipenv
```


### 1.3 Set up the Environment

Clone the GitHub repository to your local machine:
```
git clone https://github.com/emanieluu/aml_project.git
```

Navigate to the project directory:
```
cd aml_project
```

Activate the virtual environment if needed : 
```bash
pip install pipenv
pipenv shell #activate the environment 
pipenv install  #to install new packages that will be added directly to the pipfile
```

Alternatively, you can skip this and just install requirements : 
```bash
pip install -r requirements.txt
```

## 2 Data Preprocessing

We perform two types of preprocessing for the data initially provided in SMILES format:

 **Tabular Data Processing:**
   - The data is preprocessed into tabular format with feature engineering.
   - The feature engineering process is time-consuming, and for convenience, we store the tabular data in two folders: `tabular_feature_enhancement` and `tabular_preprocessed_data`. This allows us to directly access these preprocessed data when training the models.
   - `tabular_feature_enhancement` corresponds to all our 2 254 created features, and `tabular_preprocessed_data` has already passed through the first stages of our optimized pipeline, i.e. Imputer, Variance Threshold, Scaler and features extraction, leaving us with 400 features.
   - To initiate the preprocessing of these data, execute the following command:
     ```
     python -m molprop_prediction.scripts.preprocess_tabular
     ```
   Note : A "Pre-condition Violation" message may appear, but is not to be considered and does not affect preprocessing quality
    

  **Graph Data Processing:**
   - The data is preprocessed into a format suitable for graph neural networks. This process is quick and is performed just before model training when executing the training file with GIN or GAT input.


## 3 How to reproduce our results 

#### 3.1 Train and predict 
To train a model, run the following command:
```
python -m molprop_prediction.scripts.train
```
This command will execute the model training script. Follow the prompt instructions, depending if you want to train RF, GIN or GAT. Make sure you have the necessary data available and properly configured in the script. 

To get the predictions on the test and MAE score run the following command:
```
python -m molprop_prediction.scripts.predict
```

#### 3.2 Hyperparameter Optimisation 

For hyperparameters optimization, the script for RF and GIN are provided. Due to lack of time, we couldn't provide the one for GAT.
Note that these scripts take a long time to run (few hours)
```
python -m molprop_prediction.scripts.grid_search_RF
python -m molprop_prediction.scripts.grid_search_GIN
```

## 4 Folder structure

```
.
└── aml_project/
    ├── data/
    │   ├── predictions/                  # Contains the model predictions on the test set
    │   │   ├── GAT/                      # Predictions from the GAT model
    │   │   ├── GIN/                      # Predictions from the GIN model
    │   │   └── RF/                       # Predictions from the Random Forest model
    │   ├── raw_data/                     # Raw data sources
    │   ├── tabular_feature_enhancement/  # Tabular data with additional engineered features 
    │   └── tabular_preprocessed_data/    # Preprocessed tabular data (ready for training)
    ├── molprop_prediction/
    │   ├── configs/                      # Best hyperparameters for each model 
    │   │   ├── GAT/                      # Hyperparameters for the GAT model
    │   │   ├── GIN/                      # Hyperparameters for the GIN model
    │   │   └── RF/                       # Hyperparameters for the Random Forest model
    │   ├── grid_results/                 # Results from hyperparameter tuning grid search
    │   ├── models/                       # Trained machine learning models
    │   ├── scripts/                      # Main scripts for data preprocessing, model training, and optimization
    │   └── trained_models/               # Saved instances of the trained models
    ├── notebooks/                        # Jupyter notebooks for exploration and analysis
    └── reports/
        └── figures/                      # Visualizations and figures generated for reports

```
## 5 In-Depth Development Insights

- The model architecture for optimizing parameters of all the pipeline of our Random Forest model (aml_project/molprop_prediction/scripts/grid_search_RF.py), in particular the use of the hyperopt package, drew inspiration from: [GitHub link - A quantitative model for the prediction of sooting tendency from molecular structure](https://github.com/pstjohn/ysi_qsar_energy_fuels/blob/master/ysi_utils/models/setup_model.py)

- Graph data preprocessing functions were adapted from: [Blog Post - How to Turn a SMILES String into a Molecular Graph for PyTorch Geometric](https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/)

- The GIN (Graph Isomorphism Network) architecture drew inspiration from: [Blog Post - Graph Isomorphism Network](https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html)

- The remaining scripts and functionalities were meticulously crafted from scratch.
## Contributors

Claire Dechaux, François Lacarce, Emanie Luu 
