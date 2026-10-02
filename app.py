# Streamlit app for the fraud detection model
# run with:  streamlit run app.py

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.joblib")

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
columns = bundle["columns"]


def predict(raw_df):
    """takes raw transactions (Time, V1..V28, Amount) and returns fraud probability + label"""
    data = raw_df.copy()
    data[["Amount", "Time"]] = scaler.transform(data[["Amount", "Time"]])
    proba = model.predict_proba(data[columns])[:, 1]
    return proba, (proba >= threshold).astype(int)


st.title("💳 Credit Card Fraud Detector")
st.write(f"Model: **{bundle['name']}** | decision threshold: **{threshold:.2f}**")

tab1, tab2, tab3 = st.tabs(["Sample transactions", "Enter values", "Upload CSV"])

# ---- tab 1: pick one of the saved real transactions ----
with tab1:
    samples = pd.read_csv("sample_transactions.csv")
    st.write("Pick a real transaction from the test set (the model has never seen these).")
    idx = st.selectbox("Transaction", samples.index,
                       format_func=lambda i: f"#{i} - amount ${samples.loc[i, 'Amount']:.2f}")
    row = samples.loc[[idx]]
    st.dataframe(row.drop(columns="Class"))
    if st.button("Check this transaction"):
        proba, label = predict(row.drop(columns="Class"))
        if label[0] == 1:
            st.error(f"🚨 FRAUD ALERT - fraud probability {proba[0]:.1%}")
        else:
            st.success(f"✅ Looks legitimate - fraud probability {proba[0]:.1%}")
        actual = "fraud" if row["Class"].iloc[0] == 1 else "legit"
        st.caption(f"Actual label in the dataset: {actual}")

# ---- tab 2: type values in yourself ----
with tab2:
    st.write("The V1-V28 columns are anonymized PCA features, so here you can set the "
             "ones that matter most and the rest default to 0.")
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
    time_sec = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)
    top_features = ["V14", "V10", "V12", "V17", "V4"]
    values = {f: st.slider(f, -20.0, 20.0, 0.0) for f in top_features}
    if st.button("Predict"):
        entry = {c: 0.0 for c in columns}
        entry.update(values)
        entry["Amount"], entry["Time"] = amount, time_sec
        proba, label = predict(pd.DataFrame([entry]))
        if label[0] == 1:
            st.error(f"🚨 FRAUD ALERT - fraud probability {proba[0]:.1%}")
        else:
            st.success(f"✅ Looks legitimate - fraud probability {proba[0]:.1%}")

# ---- tab 3: score a whole csv ----
with tab3:
    st.write("Upload a CSV with columns Time, V1...V28, Amount (same format as the Kaggle dataset).")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        up = pd.read_csv(file)
        missing = [c for c in columns if c not in up.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            proba, label = predict(up[columns])
            out = up.copy()
            out["fraud_probability"] = proba
            out["prediction"] = ["FRAUD" if l == 1 else "legit" for l in label]
            st.write(f"{label.sum()} of {len(out)} transactions flagged as fraud")
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50))
            st.download_button("Download results", out.to_csv(index=False), "fraud_predictions.csv")
