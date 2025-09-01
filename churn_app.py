import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="USSD Customer Churn Prediction", layout="wide")

st.title("📱 USSD Customer Churn Prediction App")
st.write("This app predicts whether a USSD customer is likely to **stop using the service (churn)**.")

# Tabs for different modes
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Analytics"])

with tab1:
    st.header("🔮 Predict Customer Churn")

    # Sidebar inputs
    st.sidebar.header("Enter Customer Details")

    age = st.sidebar.slider("Age", 18, 70, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    location = st.sidebar.selectbox("Location", ["Urban", "Rural"])
    account_type = st.sidebar.selectbox("Account Type", ["Savings", "Current", "Mobile Wallet"])
    transactions_last_30d = st.sidebar.slider("Transactions (Last 30 Days)", 0, 100, 15)
    avg_transaction_value = st.sidebar.number_input("Average Transaction Value (₦)", min_value=50.0, max_value=10000.0, value=2000.0)
    failed_transactions = st.sidebar.slider("Failed Transactions", 0, 20, 2)
    sms_alerts = st.sidebar.selectbox("SMS Alerts", ["Yes", "No"])
    complaints_logged = st.sidebar.slider("Complaints Logged", 0, 10, 1)
    customer_tenure_months = st.sidebar.slider("Customer Tenure (Months)", 1, 60, 12)

    # Create dataframe for model input
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [1 if gender == "Male" else 0],
        "location": [1 if location == "Urban" else 0],
        "account_type": [0 if account_type == "Current" else (1 if account_type == "Mobile Wallet" else 2)],
        "transactions_last_30d": [transactions_last_30d],
        "avg_transaction_value": [avg_transaction_value],
        "failed_transactions": [failed_transactions],
        "sms_alerts": [1 if sms_alerts == "Yes" else 0],
        "complaints_logged": [complaints_logged],
        "customer_tenure_months": [customer_tenure_months]
    })

    # show input
    st.subheader("Customer Input Data")
    st.write(input_data)

    # Prediction
    if st.button("🔮 Predict Churn"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to **CHURN** (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Customer is likely to **STAY** (Probability: {1 - probability:.2f})")

    # Probability
        st.subheader("Prediction Probability")
        st.bar_chart({"Stay": [1 - probability], "Churn": [probability]})

with tab2:
        # Analytics Dashboard
    st.markdown("---")
    st.header("📊 Churn Analytics Dashboard")

    ussd_df = pd.read_csv(r"C:\Users\Hp\Downloads\ussd_dataset.csv")

    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    feature_importance = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax, palette='viridis')
    st.pyplot(fig)

    # Dataset explorer
    st.sidebar.subheader("Explore Dataset")
    if st.sidebar.checkbox("Show sample dataset"):
        ussd_df = pd.read_csv(r"C:\Users\Hp\Downloads\ussd_dataset.csv")
        st.subheader("📂 Sample of USSD Customer Churn Dataset")
        st.dataframe(ussd_df.head(20))

    # df = pd.read_csv("ussd_customer_churn.csv")

    col1, col2 = st.columns(2)

    # Churn rate by gender
    with col1:
        st.subheader("Churn Rate by Gender")
        churn_gender = ussd_df.groupby("gender")["churn"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x="gender", y="churn", data=churn_gender, palette="Set2", ax=ax)
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)

    # Churn rate by location
    with col2:
        st.subheader("Churn Rate by Location")
        churn_location = ussd_df.groupby("location")["churn"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x="location", y="churn", data=churn_location, palette="Set1", ax=ax)
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        # Churn by account type
        st.subheader("Churn by Account Type")
        churn_account = ussd_df.groupby("account_type")["churn"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x="account_type", y="churn", data=churn_account, palette="coolwarm", ax=ax)
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)
        
    with col4:
        # Failed transactions vs churn
        st.subheader("Failed Transact. vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="failed_transactions", data=ussd_df, palette="Pastel1", ax=ax)
        ax.set_xlabel("Churn (0=Stay, 1=Churn)")
        ax.set_ylabel("Failed Transactions")
        st.pyplot(fig)


    with st.container():
        # Customer tenure vs churn
        st.subheader("Customer Tenure vs Churn")
        fig, ax = plt.subplots()
        sns.histplot(data=ussd_df, x="customer_tenure_months", hue="churn", multiple="stack", bins=30, palette="Accent", ax=ax)
        ax.set_xlabel("Tenure (Months)")
        st.pyplot(fig)

        # Download full dataset
    st.subheader("⬇️ Download Full Dataset")
    csv_data = ussd_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as csv",
        data=csv_data,
        file_name='ussd_customer_churn.csv',
        mime="text/csv"
    )