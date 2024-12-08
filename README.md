Student Performance Prediction

Project Overview
This project aims to create a machine learning model to predict student exam scores based on various performance factors such as hours studied, motivation, attendance, parental involvement, and other relevant features. The goal is to identify key factors influencing academic performance, enabling educators to implement targeted interventions to improve student outcomes.

Additionally, this project includes visualizations and analyses, such as correlations and relationships between features like hours studied, tutoring sessions, and motivation.

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





Setup Instructions
Follow these steps to set up and run the project:

Prerequisites
Python 3.8 or higher
Required libraries (see requirements.txt)

Steps

Clone the Repository

git clone https://github.com/Sam-Little/ML_assessment_3_v2.git

cd ML_assessment_3_v2

Set Up the Environment

Install required Python libraries:
pip install -r requirements.txt

Prepare the Data
Place the dataset (StudentPerformanceFactors.csv) in the data/raw/ directory.

Usage Instructions
Run the following scripts in order to preprocess the data, train the model, and evaluate its performance.

Preprocess Data
Prepares the dataset for training by handling missing values and performing feature transformations.
python src/preprocessing/preprocess_data.py
Output: A processed dataset saved in data/processed/processed_data.csv.

Train the Model
Trains a machine learning model using the preprocessed dataset.
python src/training/train_model.py
Output: A trained model saved as models/trained_model.pkl.

Evaluate the Model
Evaluates the trained model on the dataset.
python src/evaluation/evaluate_model.py
Output: Metrics like Mean Squared Error (MSE) and R-squared value.

Explore the Data
Use the exploratory notebook for data analysis and visualization.
jupyter notebook notebooks/exploratory_analysis.ipynb

Project Goals
Identify and understand key factors influencing student performance.
Build a predictive model for exam scores.

Contact information: sam.little@students.opit.com
