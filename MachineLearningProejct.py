import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

try:
    data = pd.read_csv("*your path*")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found.")
    exit()

X = data.drop('exam_score', axis=1)
y = data['exam_score']

print("Do you want to see data information and initial plots? (yes/no)")
if input().lower() == 'yes':
    print("\nFirst 5 rows of data:")
    print(data.head())
    print("\nData information (data types, non-null counts):")
    print(data.info())
    print("\nStatistical summary of numerical features:")
    print(data.describe())
    print("\nNumber of missing values per column:")
    print(data.isnull().sum())

    all_numerical_cols_for_plot = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                                   'attendance_percentage', 'sleep_hours', 'mental_health_rating',
                                    'exam_score']

    print("\nhistograms for numerical features.")
    data[all_numerical_cols_for_plot].hist(bins=20, figsize=(18, 12), layout=(3, 3))
    plt.suptitle('Histograms of Numerical Features', y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data['exam_score'], bins=20, kde=True)
    plt.title('Distribution of Exam Scores')
    plt.xlabel('Exam Score')
    plt.ylabel('Number of Students')
    plt.show()

    categorical_features_for_plot = ['gender', 'part_time_job', 'diet_quality', 'internet_quality']

    print("\nСount plots for categorical features.")
    for col in categorical_features_for_plot:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=col, hue=col, palette='viridis', legend=False)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


else:
    print("\nData showcase and initial plots skipped.")

print("\nData cleaning: No missing values found.")

categorical_features = ['gender', 'part_time_job', 'diet_quality', 'internet_quality']
numerical_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                      'attendance_percentage', 'sleep_hours', 'mental_health_rating', 'exercise_frequency']
#lets say its method of standardization and normalization,so data that is not numbers is boolean-ish aka 1 or 0 while numerical is yea.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
print("Standardization of numerical features and One-Hot encoding of categorical features will be performed within the preprocessing pipeline.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

print("\nTraining Models with GridSearchCV")

print("\nOptimizing Random Forest Regressor")
rf_base_model = RandomForestRegressor(random_state=42)
rf_pipeline_for_gridsearch = Pipeline(steps=[('preprocessor', preprocessor),
                                              ('regressor', rf_base_model)])
rf_param_grid = {
    'regressor__n_estimators': [200, 300, 400, 500],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}
# model of pipeline,dict of parameters,cross validation folds number,method of scoring,number of CPU cores,output verbosity
rf_grid_search = GridSearchCV(rf_pipeline_for_gridsearch, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

print("Starting GridSearchCV for Random Forest.")
rf_grid_search.fit(X_train, y_train)
print("GridSearchCV for Random Forest completed.")

best_rf_pipeline = rf_grid_search.best_estimator_
print(f"\nBest parameters for Random Forest: {rf_grid_search.best_params_}")
print(f"Best R-squared on cross-validation for Random Forest: {rf_grid_search.best_score_:.2f}")

print("\nOptimizing Gradient Boosting Regressor")
gb_base_model = GradientBoostingRegressor(random_state=42)
gb_pipeline_for_gridsearch = Pipeline(steps=[('preprocessor', preprocessor),
                                              ('regressor', gb_base_model)])
gb_param_grid = {
    'regressor__n_estimators': [100, 200, 300, 400], #reduced to 400 due to time longetivity  of process
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7]
}
# model of pipeline,dict of parameters,cross validation folds number,method of scoring,number of CPU cores,output verbosity
gb_grid_search = GridSearchCV(gb_pipeline_for_gridsearch, gb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

print("Starting GridSearchCV for Gradient Boosting.")
gb_grid_search.fit(X_train, y_train)
print("GridSearchCV for Gradient Boosting completed.")

best_gb_pipeline = gb_grid_search.best_estimator_
print(f"\nBest parameters for Gradient Boosting: {gb_grid_search.best_params_}")
print(f"Best R-squared on cross-validation for Gradient Boosting: {gb_grid_search.best_score_:.2f}")

models = {
    "Random Forest Regressor": best_rf_pipeline,
    "Gradient Boosting Regressor": best_gb_pipeline
}

print("\nAnalysis of Results")

results = {}

for name, model in models.items(): # including both models look at #129 line
    print(f"\nEvaluating {name}")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (Coefficient of Determination): {r2:.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Exam Scores")
    plt.ylabel("Predicted Exam Scores")
    plt.title(f'Actual vs Predicted Exam Scores ({name})')
    plt.grid(True)
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Exam Scores")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f'Residuals Plot ({name})')
    plt.grid(True)
    plt.show()

    # this part of the code is used to extract feature importances from the trained models
    # feature importances are the measures of how important each feature is in the model
    # this is useful for understanding which features are driving the predictions
    # Random Forest Regressor and Gradient Boosting Regressor support feature importance extraction directly
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        # get the feature names from the preprocessing pipeline
        feature_names_processed = numerical_features[:]
        ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names_processed.extend(ohe_feature_names)

        # get the feature importances from the model
        importances = model.named_steps['regressor'].feature_importances_

        # create a pandas Series to hold the feature importances
        feature_importances = pd.Series(importances, index=feature_names_processed).sort_values(ascending=False)

        # plot the feature importances using a bar chart
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, hue=feature_importances.index, palette='viridis', legend=False)
        plt.title(f'Feature Importance ({name})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print(f"\n{name} does not support direct feature importance extraction.")

print("\nComparison of Model Results")
results_df = pd.DataFrame(results).T
print(results_df)
