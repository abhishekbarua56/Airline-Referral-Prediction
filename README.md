# Airline-Referral-Prediction
Classification 


## Project Description

The Airline Passenger Referral Prediction project aims to develop a machine learning model that predicts whether a passenger will refer the airline to others based on various features extracted from historical data. Understanding passenger behavior and referral patterns is crucial for enhancing customer satisfaction, improving marketing strategies, and increasing brand loyalty. This project employs classification techniques to analyze factors influencing passenger referrals, ultimately providing actionable insights for strategic decision-making.

## Objectives

- To predict the likelihood of passengers referring the airline to others.
- To identify key factors that significantly impact referral behavior.
- To provide a data-driven approach for enhancing customer experience and loyalty strategies.

## Table of Contents

1. [Technologies Used](#technologies-used)
2. [Data Sources](#data-sources)
3. [Key Steps and Methods](#key-steps-and-methods)
    - [Data Preprocessing](#data-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Model Building](#model-building)
    - [Model Evaluation](#model-evaluation)
4. [Instructions to Run the Project](#instructions-to-run-the-project)
5. [Usage Examples](#usage-examples)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact Information](#contact-information)

## Technologies Used

- **Python**: The primary programming language used for data analysis and modeling.
- **Jupyter Notebook**: An interactive environment for running Python code and visualizing results.
- **Pandas**: A powerful library for data manipulation and analysis.
- **NumPy**: A library for numerical computations that complements Pandas.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations.
- **Seaborn**: A statistical data visualization library based on Matplotlib.
- **Scikit-learn**: A library providing simple and efficient tools for data mining and machine learning.

## Data Sources

The dataset used in this project consists of historical referral data from airline passengers. It includes various features such as:

- **Passenger_ID**: Unique identifier for each passenger.
- **Age**: Age of the passenger.
- **Gender**: Gender of the passenger.
- **Flight_Distance**: Distance of the flight in miles.
- **Departure_Airport**: The airport from which the passenger departed.
- **Arrival_Airport**: The airport at which the passenger arrived.
- **Referral**: Whether the passenger referred the airline (1 for yes, 0 for no).

## Key Steps and Methods

### Data Preprocessing

- **Missing Value Handling**: Addressed missing values in relevant columns using statistical methods and imputation.
- **Encoding Categorical Variables**: Used label encoding and one-hot encoding to convert categorical variables into numerical formats suitable for modeling.
- **Feature Scaling**: Standardized features to improve model convergence and performance.

### Exploratory Data Analysis (EDA)

- Utilized visualizations (bar plots, box plots, and histograms) to identify trends and correlations between features and referrals.
- Investigated the impact of passenger demographics, flight distance, and airports on referral behavior.

### Model Building

- Trained several classification models, including:
    - **Logistic Regression**: Established a baseline model for referral prediction.
    - **Random Forest Classifier**: Used ensemble methods to improve prediction accuracy.
    - **Gradient Boosting Classifier**: Implemented boosting techniques for enhanced performance.

- Used train-test split to evaluate model performance on unseen data, ensuring a robust validation process.

### Model Evaluation

- Evaluated models using:
    - **Accuracy**: The proportion of correctly predicted referrals.
    - **F1 Score**: A measure of the model's precision and recall balance.
    - **Confusion Matrix**: Visual representation of model performance.

## Instructions to Run the Project

Clone the repository to your local machine:

```bash
git clone https://github.com/abhishekbarua56/Airline-Referral-Prediction
```

Navigate to the project directory:

```bash
cd Airline-Referral-Prediction
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Open the Jupyter Notebook (ML_Classification_project_on_Airline_Passenger_Referral_Prediction.ipynb) in Jupyter Lab or Jupyter Notebook and execute the cells in order.

## Usage Examples

To make predictions using the trained model, you can input new data into the model. Here’s a basic example of how to use the model for predictions (example provided in the notebook):

```python
import pandas as pd

# Load the trained model
import joblib
model = joblib.load('model.pkl')

# Prepare new data
new_data = pd.DataFrame({
    'Age': [30],
    'Gender': ['Male'],
    'Flight_Distance': [500],
    'Departure_Airport': ['JFK'],
    'Arrival_Airport': ['LAX']
})

# Make predictions
predicted_referral = model.predict(new_data)
print(f'Predicted Referral: {predicted_referral}')
```

## Results

The final model achieved an accuracy of [insert accuracy value] and an F1 score of [insert F1 score value], indicating a strong predictive capability. Key findings from the analysis included:

- Top Factors Influencing Referrals: Age and flight distance were found to have a significant impact on referral behavior.
- Referral Trends: Understanding demographic trends can guide targeted marketing efforts.

Visualizations and detailed metrics can be found throughout the Jupyter Notebook.

## Future Improvements

- Enhanced Feature Engineering: Further feature creation based on external data sources, such as passenger feedback or economic indicators, could improve predictions.
- Advanced Modeling Techniques: Experimenting with deep learning models for more complex relationships.
- Real-time Predictions: Develop an API for real-time referral predictions based on incoming data.

## Contributing

Contributions are welcome! If you would like to contribute to this project:

1. Fork the repository.
2. Create a new branch (e.g., feature-branch).
3. Make your changes and commit them.
4. Submit a pull request detailing your changes and their significance.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact Information

For questions, suggestions, or feedback, please feel free to reach out:

- **Name**: Abhishek Ranjit Barua
- **Email**: babi17no@gmail.com
- **GitHub**: [Abhishek's Profile](https://github.com/abhishekbarua56)
```

