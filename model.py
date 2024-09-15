from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib
import numpy as np
Hplot_df_filer_cleaned_no_outliers=pd.read_csv('Hplot_df_filer_cleaned_no_outliers.csv')
# Separate features (X) and target (y)
X = Hplot_df_filer_cleaned_no_outliers.drop(columns=['Plot__Price', 'Plot__name', 'Plot__url', 'Plot__DESC'])
y = Hplot_df_filer_cleaned_no_outliers['Plot__Price']

# Numerical columns to scale
numerical_columns = ['Plot__Beds', 'Property__Age', 'Build__Area', 'Plot__Area']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


# Initialize the GradientBoostingRegressor
model = GradientBoostingRegressor()

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200,300],                # Number of boosting stages to be run
    'max_depth': [3, 5, 7,9],                        # Maximum depth of the individual trees
    'learning_rate': [0.01, 0.1, 0.2],             # Learning rate
    'subsample': [0.6,0.8, 1.0],                       # Fraction of samples used for fitting the trees
}

# Create Grid Search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
# Function to add jitter to features
def add_jitter(X, jitter_strength=0.01):
    # Copy the original data
    X_jittered = X.copy()
    
    # Add random noise to each feature
    for column in X_jittered.columns:
        # Generate random noise
        noise = np.random.normal(loc=0, scale=jitter_strength, size=X_jittered[column].shape)
        X_jittered[column] += noise
    
    return X_jittered

# Add jitter to the features
jitter_strength = 0.003  # Adjust this value as needed
X_jittered = add_jitter(X, jitter_strength)

# Combine original and jittered data
X_combined = pd.concat([X, X_jittered], axis=0)
y_combined = pd.concat([y, y], axis=0)  # Same target values for both original and jittered data


# Split the data for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Fit Grid Search
grid_search.fit(X_train, y_train)
best_model=grid_search.best_estimator_
joblib.dump(best_model, 'gboost_model_7.pkl')
loaded_scaler = joblib.load('scaler.pkl')
print("Scaler mean:", loaded_scaler.mean_)
print("Scaler scale:", loaded_scaler.scale_)
# Predict with the best model
loaded_model=joblib.load('gboost_model_7.pkl')
y_pred = loaded_model.predict(X_test)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f'Final R-squared score on test set: {r2}')
