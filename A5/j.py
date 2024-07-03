import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class KFoldModel:
    def __init__(self, filename, target_cols):
        self.data = pd.read_csv(filename)
        self.target_cols = target_cols
        self.features = [
            'Sales in Thousands', 'Year Resale Value', 'Engine Size', 
            'Wheelbase', 'Width', 'Length', 'Curb Weight', 'Fuel Capacity', 
            'Fuel Efficiency', 'Power Perf Factor', 'Vehicle Type Car', 
            'Vehicle Type Passenger'
        ]
        self.X = self.data[self.features]
        self.y = self.data[self.target_cols]

    def train_and_evaluate(self, model, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        y_actual = []
        y_predicted = []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate the model using mean squared error
            mse = mean_squared_error(y_test, y_pred)
            fold_scores.append(mse)
            y_actual.extend(y_test.values)
            y_predicted.extend(y_pred)

        # Calculate overall metrics
        overall_mse = np.mean(fold_scores)
        r2 = r2_score(y_actual, y_predicted)
        return overall_mse, r2

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

    def visualize_predictions(self, model):
        y_pred = model.predict(self.X)
        for col in self.target_cols:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y[col], y_pred, alpha=0.5)
            plt.plot([self.y[col].min(), self.y[col].max()], [self.y[col].min(), self.y[col].max()], 'k--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs Predicted {col}')
            plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the KFoldModel with the CSV file and target columns
    model = KFoldModel('cars.csv', ['Price in Thousands', 'Horsepower'])

    # Initialize a Linear Regression model
    lr_model = LinearRegression()

    # Train and evaluate the model using k-fold cross-validation
    mse, r2 = model.train_and_evaluate(lr_model, n_splits=5)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot learning curve
    model.plot_learning_curve(lr_model)

    # Visualize predictions
    model.visualize_predictions(lr_model)
