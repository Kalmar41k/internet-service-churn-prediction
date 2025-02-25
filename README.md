# internet-service-churn-prediction

## Project Goal
The goal of this project is to predict customer churn using different machine learning models. We analyzed the data, preprocessed it, and trained four different algorithms: Random Forest, SVM, Logistic Regression, and a Deep Neural Network. Additionally, we developed a Streamlit application to interact with the models and visualize predictions, and created a Dockerfile and Docker Compose for automating the setup and deployment of the application.

## What has been done:
### 1. Exploratory Data Analysis (EDA)
- Olha conducted a detailed analysis of the distribution of features and the target variable, and identified anomalous values in the dataset using the describe() method. For example, she found a negative value in the subscription_age column.
- Oleksandr performed a correlation analysis, but did not find any strongly correlated features that could negatively affect the model.
- Vitalii analyzed the missing values and found out that the remaining_contract (21572/72274) columns have a significant correlation with other important columns (id, churn). For this column, it was decided to replace the missing values with zeros. For the download_avg and upload_avg columns, the missing values were replaced by the median, as they had almost no correlation with the other features.

### 2. Data Preprocessing
Vitalii was in charge of data processing:
- Removed the anomalous values identified by Olha,
- Filled in the missing values according to the chosen strategy,
- Excluded the id column as it is not useful for modeling,
- Data standardization was planned at the model training stage to avoid changes in the data distribution at the processing stage.
- The processed dataset was saved in CSV format for further use in the model training stage.

### 3. Model Development
All team members used the GridSearchCV method to optimize the hyperparameters of the models:
- *LogisticRegression, a model trained by Olha:*
  - Precision: 0.89 (class 0), 0.85 (class 1)
  - Recall: 0.81 (class 0), 0.92 (class 1)
  - F1-score: 0.85 (class 0), 0.88 (class 1)
- *Random Forest, model trained by Oleksandr:*
  - Precision: 0.92 (class 0), 0.95 (class 1)
  - Recall: 0.95 (class 0), 0.93 (class 1)
  - F1-score: 0.93 (class 0), 0.94 (class 1)
*It was the most accurate model by all metrics and showed the best results.*
- *SVM (Support Vector Machine), a model trained by Olha:*
  - Precision: 0.91 (class 0), 0.92 (class 1)
  - Recall: 0.91 (class 0), 0.93 (class 1)
  - F1-score: 0.91 (class 0), 0.92 (class 1)
- *Deep Neural Network, model trained by Vitalii:*
  - Precision: 0.92 (class 0), 0.94 (class 1)
  - Recall: 0.92 (class 0), 0.93 (class 1)
  - F1-score: 0.92 (class 0), 0.93 (class 1)

### 4. Integration and Result Output
Oleksandr created an interactive interface in Streamlit, where all the models for forecasting were integrated, and the function of standardizing the training set was implemented. After testing the application, there were no errors and it functioned successfully when entering data for forecasting.

### 5. Docker Integration
To facilitate deployment, Vitalii created a Dockerfile containing:
- Added the necessary dependencies in requirements.txt,
- Created and configured the Dockerfile, docker-compose.yml and .dockerignore files,
- Tested the application in the Docker environment, which successfully launched the application with all dependencies and configurations.

## Authors
- Kalmar41k - Vitalii
- OlhaYastrebova - Olha
- Oleksandr0210 - Oleksandr

## How to Run the Project
To run this project in Docker, follow the steps below:

### Prerequisites:
- Docker must be installed on your machine.

### Steps to Run:
1. **Clone the Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Kalmar41k/internet-service-churn-prediction.git
   cd customer-churn-prediction
   ```
2. **Build the Docker Image**: 
   In the root of the repository, build the Docker image by running:
   ```bash
   docker-compose build
   ```
3. **Run the Docker Containers**:
   Start the containers with Docker Compose:
   ```bash
   docker-compose up
   ```
4. **Access the Application**:
   Once the containers are up and running, open your web browser and go to http://localhost:8501. You should see the Streamlit app where you can interact with the models and make predictions.

### Docker Commands Breakdown:
- docker-compose build: Builds the Docker image based on the Dockerfile.
- docker-compose up: Starts the application in Docker, running the app in the container.
- docker-compose down: Stops the containers when needed.

## Notes:
- If you wish to stop the application, you can press Ctrl+C or run docker-compose down to stop the containers.
- Ensure that all dependencies are listed in the requirements.txt file and correctly included in the Docker setup.

### How It Works:
1. **Dockerfile**: This file is responsible for defining the environment the application will run in. It installs necessary dependencies like TensorFlow, scikit-learn, Streamlit, etc.
2. **Docker Compose**: This file automates the process of building and running the Docker containers. It ensures that all the dependencies are set up and the application can be run with a single command.

## Happy Predicting!
