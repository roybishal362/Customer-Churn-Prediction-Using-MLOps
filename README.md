# Customer Churn Prediction - MLOps Project

## Project Overview
This project aims to predict customer churn using machine learning with an MLOps-ready approach. While the CI/CD pipeline is not currently implemented due to lack of an Amazon Cloud subscription, the project has been structured with Docker and other tools, making future integration seamless.

## Key Features
- **Exploratory Data Analysis (EDA):** Understand data patterns and customer behavior.
- **Feature Engineering:** Enhance model performance through feature creation.
- **Machine Learning Models:** Predict churn with models such as Random Forest achieving 99% accuracy.
- **MLOps Setup:** The project is ready for CI/CD pipeline integration, containerized using Docker.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools:** Docker, GitHub Actions

## Modeling Process
- **Data Preprocessing:** Handle missing data, encoding, and scaling.
- **EDA:** Visualize and analyze customer churn patterns.
- **Modeling:** Build models like Random Forest.
- **Evaluation:** Evaluate performance using accuracy and F1-score.

## Deployment Setup
- **Containerization:** Docker is used to ensure environment consistency.
- **CI/CD Setup:** Prepared for future deployment using AWS ECR once available.

## How to Run the Project Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/roybishal362/Customer-Churn-Prediction-Using-MLOps.git
