import joblib
from flask import Flask, request, jsonify

#load in model
log_reg = joblib.load('final_model.pkl')

# create flask app
app = Flask('pima')

@app.route('/predict', methods=['POST'])
def predict():
    person= request.get_json()

    from sklearn.feature_extraction import DictVectorizer
    dv = DictVectorizer(sparse= False)

    X = dv.fit_transform([person])

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    X_prep = pipeline.fit_transform(X)

    y_pred = log_reg.predict_proba(X_prep)[0, 1]
    diabetic = y_pred >= 0.5

    result = {
        'diabetic_probability': float(y_pred),
        'diabetic': bool(diabetic)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)