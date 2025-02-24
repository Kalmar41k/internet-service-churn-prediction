# internet-service-churn-prediction

## Project Goal
The goal of this project is to predict customer churn using different machine learning models. We analyzed the data, preprocessed it, and trained four different algorithms: Random Forest, SVM, Logistic Regression, and a Deep Neural Network. Additionally, we developed a Streamlit application to interact with the models and visualize predictions, and created a Dockerfile and Docker Compose for automating the setup and deployment of the application.

## What has been done:
- **Exploratory Data Analysis (EDA)**: We performed an in-depth analysis of the dataset to understand the key features affecting customer churn.
- **Data Preprocessing**: Data was cleaned and transformed to ensure the models could learn effectively.
- **Model Development**: We trained four machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Deep Neural Network (DNN)
- **Integration and Result Output**: We integrated the trained models into a Streamlit application, allowing users to interact with the models and predict whether a customer will churn or not.
- **Dockerfile**: We created a Dockerfile to containerize the application for easy deployment.
- **Docker Compose**: We used Docker Compose to automate the setup and execution of the app in Docker.

## Authors
- Kalmar41k
- OlhaYastrebova
- Oleksandr0210

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

## Notes:
- If you wish to stop the application, you can press Ctrl+C or run docker-compose down to stop the containers.
- Ensure that all dependencies are listed in the requirements.txt file and correctly included in the Docker setup.

### How It Works:
1. **Dockerfile**: This file is responsible for defining the environment the application will run in. It installs necessary dependencies like TensorFlow, scikit-learn, Streamlit, etc.
2. **Docker Compose**: This file automates the process of building and running the Docker containers. It ensures that all the dependencies are set up and the application can be run with a single command.

Happy Predicting!