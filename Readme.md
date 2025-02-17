### Price Prediction with Continuous Integration & Deployment Using ZenML
This project is not just about developing a machine learning model for price prediction but also focuses on implementing a Continuous Integration (CI) and Continuous Deployment (CD) system using ZenML. The goal was to create a robust, automated, and scalable pipeline for model training, evaluation, and deployment.

##### Key Steps & Implementation

###### Exploratory Data Analysis (EDA) and Preprocessing

* Developed a structured Python script to conduct an in-depth analysis of the dataset.
* Designed modular classes for each analytical step, improving code reusability and maintainability.
* Implemented various data inspection and analysis techniques:
    * Data Extraction: Loaded and prepared the dataset.
    * Data Inspection: Checked for data integrity, missing values, and inconsistencies.
    * Missing Value Analysis: Applied imputation techniques to handle missing values.
    * Univariate Analysis: Performed descriptive statistics and visualized distributions.
    * Bivariate Analysis: Explored relationships between different variables using correlation matrices and scatter plots.
    * Heatmap Visualization: Used Seaborn to generate heatmaps for better feature correlation understanding.

###### Machine Learning Model Development

Structured the machine learning pipeline into separate Python scripts, each handling a key aspect of the model development process:
    * Data Splitting: Split the dataset into training, validation, and test sets.
    * Feature Engineering: Applied transformations such as scaling, encoding, and feature selection.
    * Handling Missing Values: Implemented appropriate imputation techniques (mean, median, KNN imputation).
    * Model Building: Trained multiple models (Linear Regression, Random Forest, XGBoost) to compare performance.
    * Model Evaluation: Measured performance using RMSE, R², and MAPE, selecting the best-performing model.

###### CI/CD Pipeline with ZenML for Automation

Integrated ZenML to automate the ML workflow using CI/CD pipelines, ensuring efficient training, validation, and deployment.
* Developed two key scripts:
    * run_pipeline.py: This script orchestrates the ML pipeline by executing all steps, from data ingestion to model training and validation.
    * run_deployment.py: This script deploys the trained model, enabling real-time inference on new incoming data.

Ensured reproducibility and scalability of the pipeline using ZenML’s orchestrators and artifact store, making it easier to retrain and update models dynamically.

###### Outcome & Impact
Established a fully automated CI/CD pipeline for continuous training and deployment of price prediction models.
Improved model retraining efficiency by automating data ingestion, preprocessing, and validation.
Enabled seamless deployment of ML models, ensuring real-time price predictions with high accuracy.