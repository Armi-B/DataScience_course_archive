import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import math
random_states = [42, 123, 456, 789]
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

class MultivariateLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def prepare_data(self, data, X, Y):
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.X_train_final = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
        self.X_test_final = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
                
    def fit(self):
        self.theta = np.zeros(self.X_train_final.shape[1])
        self.cost_history = []

        for i in range(self.n_iterations):
            y_pred = np.dot(self.X_train_final, self.theta)
            error = y_pred - self.y_train
            cost = np.mean(error**2) / 2
            self.cost_history.append(cost)
            gradient = np.dot(self.X_train_final.T, error) / len(self.y_train)
            self.theta -= self.learning_rate * gradient

    def predict(self):
        return np.dot(self.X_test_final, self.theta)
    
    def evaluate(self, y_name, y_pred):
        rmse = math.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        scores_df = pd.DataFrame(columns=['RMSE', 'R2_score'])
        scores_df.loc[0] = [rmse, r2]
        return scores_df
    
    def show_prediction_result(self, y_pred, y_name):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, color=PALETTE[6]) 
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color=PALETTE[1], linestyle='--') 
        plt.xlabel('Actual ' + y_name)
        plt.ylabel('Predicted ' + y_name)
        plt.title('Actual vs. Predicted ' + y_name)
        plt.show()

    def show_learning_curve(self, y_name):
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.n_iterations), self.cost_history, color=PALETTE[0])
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent Learning Curve (' + y_name + ')')
        plt.show()
    
    def plot_accuracy_across_random_states(self, X, Y, random_states):
        r2_scores = []
        for state in random_states:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=state)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_final = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
            X_test_final = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
            
            self.X_train_final = X_train_final
            self.X_test_final = X_test_final
            self.y_train = y_train
            self.y_test = y_test
            
            self.fit()
            y_pred = self.predict()
            r2 = r2_score(self.y_test, y_pred)
            r2_scores.append(r2)
        
        plt.figure(figsize=(10, 6))
        plt.bar([str(state) for state in random_states], r2_scores, color=PALETTE[2])
        plt.xlabel('Random State')
        plt.ylabel('R2 Score')
        plt.title('R2 Score Across Different Random States')
        plt.show()

data = pd.read_csv("cars.csv")
X = data[['Sales in Thousands', 'Year Resale Value', 'Engine Size', 'Wheelbase', 'Width', 'Length', 'Curb Weight', 'Fuel Capacity', 'Fuel Efficiency', 'Power Perf Factor', 'Vehicle Type Car', 'Vehicle Type Passenger']]
y_price = data['Price in Thousands']
y_horsepower = data['Horsepower']

model_price = MultivariateLinearRegression(learning_rate=0.1, n_iterations=150)



model_price.plot_accuracy_across_random_states(X, y_price, random_states)
