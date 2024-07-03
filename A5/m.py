import pandas as pd
import numpy as np
from sklearn.model_selection import KFold ,learning_curve
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class KFoldModel:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.features = [
            'Sales in Thousands', 'Year Resale Value', 'Engine Size', 
            'Wheelbase', 'Width', 'Length', 'Curb Weight', 'Fuel Capacity', 
            'Fuel Efficiency', 'Power Perf Factor', 'Vehicle Type Car', 
            'Vehicle Type Passenger'
        ]
        self.X = self.data[self.features]
        self.y_price = self.data['Price in Thousands']
        self.y_horsepower = self.data['Horsepower']

    def train_and_evaluate(self, model, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        y_price_actual = []
        y_price_predicted = []
        y_horsepower_actual = []
        y_horsepower_predicted = []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_price_train, y_price_test = self.y_price.iloc[train_idx], self.y_price.iloc[test_idx]
            y_horsepower_train, y_horsepower_test = self.y_horsepower.iloc[train_idx], self.y_horsepower.iloc[test_idx]

            model.fit(X_train, y_price_train)
            y_price_pred = model.predict(X_test)
            fold_scores.append(model.score(X_test, y_price_test))
            y_price_actual.extend(y_price_test)
            y_price_predicted.extend(y_price_pred)

            model.fit(X_train, y_horsepower_train)
            y_horsepower_pred = model.predict(X_test)
            y_horsepower_actual.extend(y_horsepower_test)
            y_horsepower_predicted.extend(y_horsepower_pred)

        return fold_scores, y_price_actual, y_price_predicted, y_horsepower_actual, y_horsepower_predicted

    def plot_learning_curve(self, model, train_sizes=np.linspace(0.1, 1.0, 5), n_splits=5):
        plt.figure(figsize=(8, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X, self.y, train_sizes=train_sizes, cv=n_splits, scoring='neg_mean_squared_error'
        )
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, test_scores_mean, label='Validation error')
        plt.xlabel('Training examples')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    def visualize_predictions(self, actual, predicted, target_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted {target_name}')
        plt.show()


model = KFoldModel('cars.csv')

# Initialize a Linear Regression model
lr_model = LinearRegression()

# Train and evaluate the model for Price in Thousands
_, y_price_actual, y_price_predicted, _, _ = model.train_and_evaluate(lr_model)

# Visualize predictions for Price in Thousands
model.visualize_predictions(y_price_actual, y_price_predicted, 'Price in Thousands')

# Train and evaluate the model for Horsepower
_, _, _, y_horsepower_actual, y_horsepower_predicted = model.train_and_evaluate(lr_model)

# Visualize predictions for Horsepower
model.visualize_predictions(y_horsepower_actual, y_horsepower_predicted, 'Horsepower')
