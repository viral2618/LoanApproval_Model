import joblib
import numpy as np
import pandas as pd
from flask import render_template,request,redirect,url_for,session,flash,Flask

app=Flask(__name__)

model=joblib.load('Random_Forest_Model.pkl')
scaler=joblib.load('Scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def prediction():
    try:
        features = [float(x) for x in request.form.values()]
            # Step 2: Define column names (same order as training data)
        num_cols = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
                'loan_to_income', 'total_assets', 'Loan_asset', 'assets_to_loan',
                'monthly_income', 'monthly_emi', 'emi_to_income_ratio',
                'income_per_dependent']

    # Step 3: Convert to DataFrame
        df_input = pd.DataFrame([features], columns=num_cols)

    # Step 4: Apply log1p on the same columns used during training
        log_cols = ['income_annum', 'loan_amount', 'residential_assets_value',
                'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

        for col in log_cols:
            df_input[col] = df_input[col].apply(lambda x: x if x >= 0 else 0)
            df_input[col] = np.log1p(df_input[col])

    # Step 5: Apply same scaler transformation
        scaled_input = scaler.transform(df_input[num_cols])

    # Step 6: Predict using the trained model
        prediction = model.predict(scaled_input)[0]

    # Step 7: Interpret prediction
        status = 'Loan Approved' if prediction == 1 else 'Loan Not Approved'

        return render_template('prediction.html', prediction_text=f'Loan Status: {status}')

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)