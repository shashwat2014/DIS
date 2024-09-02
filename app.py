from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  # Added for p-value extraction from the logistic regression model
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the files are present in the request
        if "lead_data" not in request.files or "conversion_data" not in request.files:
            return "Error: Please upload both datasets."

        # Read the input files from the form
        lead_data = pd.read_excel(request.files["lead_data"])
        conversion_data = pd.read_excel(request.files["conversion_data"])

        # Step 3: Check if the number of rows in the first dataset is equal to the number of rows in the third dataset
        if lead_data.shape[0] != conversion_data.shape[0]:
            raise ValueError("Row count doesn't match between the first and third datasets.")

        # Step 4: Convert all categorical columns in the first dataset into multiple columns using one-hot encoding
        categorical_columns = lead_data.select_dtypes(include=['object']).columns.tolist()
        # Use pd.get_dummies to create one-hot encoded variables and ensure they are integers
        lead_data_encoded = pd.get_dummies(lead_data, columns=categorical_columns, drop_first=True).astype(int)
        
        
        # Step 5: Initial Logistic Regression Model with All Features
        # Combine the lead data with the conversion data to include the dependent variable in the analysis
        initial_data = lead_data_encoded.copy()
        initial_data['Conversion'] = conversion_data['Conversion']
        
        # Prepare data for logistic regression
        X = initial_data.drop('Conversion', axis=1)  # Independent variables
        
        # Convert all columns to numeric, forcing errors to NaN if conversion fails
        X = X.apply(pd.to_numeric, errors='coerce')  
        X = sm.add_constant(X)  # Add constant term for statsmodels
        
        y = pd.to_numeric(initial_data['Conversion'], errors='coerce')  # Ensure y is numeric
        
        # Check for missing values and handle them (e.g., drop or fill)
        if X.isnull().any().any() or y.isnull().any():
            print("Missing values detected. Handling missing values by dropping rows with NaNs.")
            # Drop rows with NaN values in either X or y
            X = X.dropna()
            y = y[X.index]  # Align y with X after dropping rows
        
        def calculate_bic(X, y):
            model = sm.Logit(y, X).fit(disp=0)
            bic_value = model.bic  # Extract BIC value from the fitted model
            return bic_value, model
        
        # Initial feature set: start with all features
        current_features = list(X.columns)
        best_bic, best_model = calculate_bic(X[current_features], y)
        
        
        # Perform backward elimination based on BIC
        improvement = True
        
        while improvement and len(current_features) > 1:
            bic_values = {}
            # Try removing each feature to see if BIC improves
            for feature in current_features:
                features_to_test = [f for f in current_features if f != feature]
                try:
                    bic_value, _ = calculate_bic(X[features_to_test], y)
                    bic_values[feature] = bic_value
                except Exception as e:
                    # Handle potential fitting issues with certain feature subsets
                    continue
        
            # Identify the feature whose removal gives the best BIC
            feature_to_remove = min(bic_values, key=bic_values.get)
            min_bic = bic_values[feature_to_remove]
        
            # Check if removing this feature improves BIC
            if min_bic < best_bic:
                current_features.remove(feature_to_remove)
                best_bic = min_bic
            else:
                improvement = False
        
        # Final model with the best BIC
        final_model = best_model
        print(f"Final BIC: {best_bic}")
        print(f"Selected features: {current_features}")
        
        # Create the final dataset with the selected significant columns
        final_data = lead_data_encoded[current_features[1:]]  # Exclude 'const'
        
        # Step 7: Create a final logistic regression model using the selected significant features
        model = LogisticRegression()
        model.fit(final_data, conversion_data['Conversion'])


        # Step 7: Handle new lead data upload
        if "new_leads" not in request.files:
            return "Error: Please upload the new leads dataset."

        new_leads = pd.read_excel(request.files["new_leads"])

        # Check if columns in new leads dataset are same as those from the first dataset
        if set(new_leads.columns) != set(lead_data.columns):
            return "Error: Columns don't match in the new leads dataset."

        # Step 8: Predict the probability of conversion for each new lead
        new_leads_encoded = pd.get_dummies(new_leads, columns=categorical_columns, drop_first=True).astype(int)

        # Ensure the columns match the trained model's columns
        missing_cols = set(final_data.columns) - set(new_leads_encoded.columns)
        for col in missing_cols:
            new_leads_encoded[col] = 0

        # Align the column order
        new_leads_encoded = new_leads_encoded[final_data.columns]

        # Predict probabilities
        predictions = model.predict_proba(new_leads_encoded)[:, 1]
        new_leads['Conversion_Probability'] = predictions

        # Step 9: Rank leads based on the predicted conversion probability
        new_leads['Rank'] = new_leads['Conversion_Probability'].rank(ascending=False)
        new_leads_sorted = new_leads.sort_values(by='Rank')

        # Output the ranked leads as HTML
        return new_leads_sorted.to_html()

    return render_template("index.html")

import os

if __name__ == "__main__":
    use_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=use_reloader)
