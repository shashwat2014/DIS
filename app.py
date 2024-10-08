from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  # Added for p-value extraction from the logistic regression model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
        
        # Step 2: Standardize the data for Lasso regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Step 3: Split data into training and testing sets to evaluate the model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Step 4: Apply Lasso Regularization (L1) for feature selection
        lasso_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        lasso_model.fit(X_train, y_train)
        
        # Step 5: Extract the coefficients and identify significant features
        coefficients = lasso_model.coef_.flatten()
        significant_indices = np.where(coefficients != 0)[0]  # Indices of non-zero coefficients
        significant_columns = X.columns[significant_indices]
        
        print(f"Significant features selected by Lasso: {significant_columns.tolist()}")
        
        # Step 6: Create the final dataset with only the significant columns
        final_data = X[significant_columns]

        
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

        # Save the result to a CSV file
        output_file = 'output.csv'
        new_leads_sorted.to_csv(output_file, index=False)

        # Convert DataFrame to HTML
        result_html = new_leads_sorted.to_html(classes='table table-striped table-bordered')

        return render_template('index.html', result_html=result_html, download_link=output_file)


    return render_template("index.html")

import os

if __name__ == "__main__":
    use_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=use_reloader)
