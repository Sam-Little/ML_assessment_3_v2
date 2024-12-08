Student Performance Prediction

Project Overview
This project aims to create a machine learning model to predict student exam scores based on various performance factors such as hours studied, motivation, attendance, parental involvement, and other relevant features. The goal is to identify key factors influencing academic performance, enabling educators to implement targeted interventions to improve student outcomes.

Additionally, this project includes visualizations and analyses, such as correlations and relationships between features like hours studied, tutoring sessions, and motivation.

Setup Instructions
Follow these steps to set up and run the project:

Prerequisites
Python 3.8 or higher.
Install Jupyter Notebook (optional but recommended).
Install required libraries:

pip install -r requirements.txt
Run the Project

Clone the Repository

git clone https://github.com/Sam-Little/ML_assessment_3_v2.git
cd ML_assessment_3_v2

Work Flow: 
Prepare the Data
Ensure the raw dataset (StudentPerformanceFactors.csv) is located in:
data/raw/StudentPerformanceFactors.csv

Run the Full Workflow Run the following scripts in sequence to see the full workflow:

Preprocess the Data:
python src/preprocessing/preprocess_data.py
Output: Processed dataset saved to:data/processed/processed_data.csv

Split the Data:
python data/splits/split_data.py
Output: Training and testing splits saved to:data/splits/data_splits.pkl

Train Models: Train both models (Linear Regression and Random Forest):
python src/training/train_linear_regression.py
python src/training/train_random_forest.py
Output:
models/linear_model.pkl: Trained Linear Regression model.
models/random_forest_model.pkl: Trained Random Forest model.


Evaluate Models:
python src/evaluation/evaluate_models.py
Output: Evaluation metrics (e.g., Mean Squared Error, R-squared) printed in the terminal.


Explore the Data and Results Open the Jupyter Notebooks for analysis and visualizations:
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/evaluate_models.ipynb






Directory Structure
The project is organized as follows:


ML_assessment_3_v2/
├── data/
│   ├── processed/          # Contains preprocessed data
│   │   └── processed_data.csv
│   ├── raw/                # Contains the original dataset
│   │   └── StudentPerformanceFactors.csv
│   ├── splits/             # Contains train/test splits
│       └── data_splits.pkl
├── models/                 # Saved machine learning models
│   ├── linear_model.pkl    # Trained Linear Regression model
│   └── random_forest_model.pkl # Trained Random Forest model
├── notebooks/              # Jupyter notebook(s) for analysis and evaluation
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── evaluate_models.ipynb # Model evaluation and visualization
├── src/                    # Source code for the project
│   ├── evaluation/         # Scripts for model evaluation
│   ├── preprocessing/      # Scripts for data preprocessing
│   ├── training/           # Scripts for training models
│   │   ├── train_linear_regression.py # Linear Regression training
│   │   └── train_random_forest.py    # Random Forest training
├── .gitignore              # Files and directories to ignore in Git
├── README.md               # Project overview and instructions
├── requirements.txt        # Required dependencies



Contact Details: sam.little@opit.students.com