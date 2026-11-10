
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.title("📩 Spam and Ham Detection")
st.write("Naive Bayes - Alpha Hyperparameter Tuning")

# Upload dataset
file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select columns
    text_col = st.selectbox("Select Text Column", df.columns)
    target_col = st.selectbox("Select Target Column", df.columns)

    if text_col == target_col:
        st.error("Select different columns for text and target.")
        st.stop()

    # Clean missing values
    data = df[[text_col, target_col]].dropna()

    X = data[text_col].astype(str)
    y = data[target_col]

    st.subheader("Class Distribution")
    st.write(y.value_counts())
    st.bar_chart(y.value_counts())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Vectorization
    cv = CountVectorizer()

    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    # Alpha selection
    st.subheader("Hyperparameter Tuning")

    alpha = st.select_slider(
        "Select Alpha Value",
        options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        value=1.0
    )

    fit_prior = st.checkbox(
        "Learn class prior probabilities",
        value=True
    )

    # Model
    model = MultinomialNB(
        alpha=alpha,
        fit_prior=fit_prior
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("Model Performance")

    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    st.write("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    st.dataframe(
        pd.DataFrame(
            cm,
            index=model.classes_,
            columns=model.classes_
        )
    )

    st.write("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.write("### Current Model Settings")
    st.write("Alpha:", alpha)
    st.write("Fit Prior:", fit_prior)

    # Custom message prediction
    st.subheader("Test Your Own Message")

    message = st.text_input("Enter an SMS message")

    if st.button("Predict"):
        if message.strip():
            msg_vector = cv.transform([message])
            prediction = model.predict(msg_vector)[0]

            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter a message.")
