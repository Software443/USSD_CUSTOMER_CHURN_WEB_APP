📊 Customer Churn Prediction App

A machine learning web application built with Streamlit and Python to predict customer churn. The app helps businesses identify customers who are likely to stop using their services so they can take proactive retention measures.

🚀 Features

Interactive web interface using Streamlit

Upload customer data (CSV) for prediction

Built-in exploratory data analysis (EDA)

Visualizations (distribution plots, churn rates, etc.)

Machine learning model to predict churn probability

Option to download prediction results

🛠️ Tech Stack

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

Streamlit (for the web interface)

Joblib/Pickle (for model persistence)

customer-churn-app/
│
├── churn_app.py           # Main Streamlit app
├── churn_model.pkl        # Saved ML model
├── requirements.txt       # Dependencies
├── data/                  # Sample datasets
│   └── customers.csv
├── Customer Churn.ipynb   # EDA and training notebooks
└── README.md              # Project documentation
