mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the load time series and encoded time covariates dataset
load_data = pd.read_csv("/Users/macbookpro2017/Downloads/load-and-time-REN-2018-2019_1/series_2018_2020.csv")
covariates_data = pd.read_csv("/Users/macbookpro2017/Downloads/load-and-time-REN-2018-2019_1/time_covariates_2018_2020.csv")

# Merge load data and covariates data
merged_data = pd.merge(load_data, covariates_data, on="Data e Hora")

# Convert the date strings in the merged_data to datetime objects
merged_data["Data e Hora"] = pd.to_datetime(merged_data["Data e Hora"], format="%Y-%m-%d %H:%M:%S")

# Separate features and target variables
X = merged_data[['year', 'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos',
                 'minute_sin', 'minute_cos', 'dayofweek_sin', 'dayofweek_cos', 'weekofyear_sin', 'weekofyear_cos',
                 'holidays']]
y = merged_data["Consumo"]

# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

# Perform dimensionality reduction using PCA
pca = PCA(n_components=14)
X_pca = pca.fit_transform(scaled_data)

# Perform clustering using K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_pca)
clusters = kmeans.predict(X_pca)

# Print explained variance ratio of PCA
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot clustering results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering Results')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(0.8 * len(merged_data))
X_train = X_pca[:train_size]
y_train = y[:train_size]
X_test = X_pca[train_size:, :]
y_test = y[train_size:]

# Define and train machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.1),  # Adjust alpha value if needed
    "Lasso Regression": Lasso(alpha=0.1),  # Adjust alpha value if needed
    "Support Vector Regression": SVR(),
    "Nu Support Vector Regression": NuSVR(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Multi-layer Perceptron": MLPRegressor()
}

# Define parameter grids for GridSearchCV
param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.01, 0.1, 1.0]},  # Adjust alpha values if needed
    "Lasso Regression": {"alpha": [0.01, 0.1, 1.0]},  # Adjust alpha values if needed
    "Support Vector Regression": {"C": [1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]},
    "Nu Support Vector Regression": {"C": [1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]},
    "Random Forest": {"n_estimators": [100, 200, 300]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300]},
    "Multi-layer Perceptron": {"hidden_layer_sizes": [(100,), (100, 50), (100, 100)]}
}

best_models = {}

# Perform grid search for each model
for model_name, model in models.items():
    print("Training", model_name)
    grid_search = GridSearchCV(model, param_grid=param_grids[model_name], scoring="neg_mean_squared_error",
                               cv=TimeSeriesSplit(n_splits=3))
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print()

# Evaluate the models on the test set
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(model_name, "MSE:", mse)
    print(model_name, "MAE:", mae)
    print()
    
# Plot the forecast on the test set
    plt.plot(y_test.index, y_test.values, label="True")
    plt.plot(y_test.index, y_pred, label=model_name)
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.title("Model Forecast on Test Set - " + model_name)
    plt.legend()
    plt.show()

# Perform cross-validation for each model
for model_name, model in best_models.items():
    scores = cross_val_score(model, X_pca, y, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3))
    mse_scores = -scores
    mae_scores = np.sqrt(mse_scores)
    print(model_name, "Cross-Validation MSE:", mse_scores)
    print(model_name, "Cross-Validation MAE:", mae_scores)
    print("Mean CV MSE:", np.mean(mse_scores))
    print("Mean CV MAE:", np.mean(mae_scores))
    print()

# Train SARIMAX model
sarimax_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
sarimax_model_fit = sarimax_model.fit()

# Forecast using SARIMAX model
y_pred_sarimax = sarimax_model_fit.forecast(steps=len(y_test))

# Calculate FSS
def calculate_fss(y_true, y_pred, y_pred_sarimax):
    fss = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_pred_sarimax) ** 2)
    return fss

fss = calculate_fss(y_test, y_pred, y_pred_sarimax)
print("FSS:", fss)

# Find the best model based on the forecast
best_model_name = None
best_mse = float("inf")
for model_name, model in best_models.items():
    forecast = model.predict(X_test)
    mse = mean_squared_error(y_test, forecast)
    if mse < best_mse:
        best_mse = mse
        best_model_name = model_name

# Plot the forecast of the best model
best_model = best_models[best_model_name]
forecast = best_model.predict(X_test)
plt.plot(y_test.index, y_test.values, label="True")
plt.plot(y_test.index, forecast, label=best_model_name)
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Best Model Forecast - " + best_model_name)
plt.legend()
plt.show()

# Analysis of machine learning models
for model_name, model in best_models.items():
    model.fit(X_pca, y)  # Train the model on the entire dataset
    y_pred_model = model.predict(X_pca)  # Generate predictions for the entire dataset

    # Evaluate the model on the entire dataset
    mse = mean_squared_error(y, y_pred_model)
    mae = mean_absolute_error(y, y_pred_model)
    print(model_name, "MSE:", mse)
    print(model_name, "MAE:", mae)

    # Plot the forecast
    plt.plot(merged_data["Data e Hora"], merged_data["Consumo"], label="True")
    plt.plot(merged_data["Data e Hora"], y_pred_model, label=model_name)
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.title("Model Forecast - " + model_name)
    plt.legend()
    plt.show()

# SARIMAX forecast
sarimax_forecast = sarimax_model_fit.forecast(steps=len(y_test))

# Evaluate SARIMAX model
sarimax_mse = mean_squared_error(y_test, sarimax_forecast)
sarimax_mae = mean_absolute_error(y_test, sarimax_forecast)
print("SARIMAX MSE:", sarimax_mse)
print("SARIMAX MAE:", sarimax_mae)

# Plot SARIMAX forecast
plt.plot(y_test.index, y_test.values, label="True")
plt.plot(y_test.index, sarimax_forecast, label="SARIMAX")
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("SARIMAX Forecast")
plt.legend()
plt.show()

# Plot the forecast of the best model and SARIMAX
plt.plot(y_test.index, y_test.values, label="True")
plt.plot(y_test.index, forecast, label=best_model_name)
plt.plot(y_test.index, sarimax_forecast, label="SARIMAX")
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Best Model vs. SARIMAX Forecast")
plt.legend()
plt.show()