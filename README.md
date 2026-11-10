# 📩 Intelligent SMS Spam & Ham Detection Using Naive Bayes

## 📌 Project Overview
This project is an interactive **SMS Spam and Ham Detection System** developed using Machine Learning and Streamlit. It classifies text messages into two categories: **Spam** (unwanted or promotional messages) and **Ham** (legitimate messages).

The application uses the **Multinomial Naive Bayes algorithm** with CountVectorizer to convert text messages into numerical features and perform classification.

## 🚀 Features
- **Text Classification:** Identifies whether an SMS is Spam or Ham.
- **Text Vectorization:** Uses CountVectorizer to convert text into numerical features.
- **Machine Learning:** Implements Multinomial Naive Bayes.
- **Hyperparameter Tuning:** Experiments with different Alpha values to improve model performance.
- **Model Evaluation:** Displays Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- **Interactive Interface:** Built using Streamlit for real-time message predictions.
- **Model Persistence:** Saves the trained model and vectorizer together using a single Pickle file through Joblib.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- Multinomial Naive Bayes
- CountVectorizer
- Streamlit
- Joblib

## ⚙️ Workflow
1. Load and explore the SMS dataset.
2. Handle missing values.
3. Separate input messages and target labels.
4. Split the dataset into training and testing sets.
5. Convert text into numerical features using CountVectorizer.
6. Train the Multinomial Naive Bayes model.
7. Experiment with different Alpha values.
8. Evaluate model performance.
9. Save the trained pipeline.
10. Deploy the application using Streamlit.

## ▶️ How to Run the Project

**1. Clone the repository**
```bash
git clone YOUR_REPOSITORY_URL
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**
```bash
streamlit run app.py
```
![Screenshot 2 ](images/"C:\Users\kudip\OneDrive\Pictures\Screenshots\Screenshot 2026-11-10 160922.png")

## 🎯 Objective
The main objective is to build a simple and interactive machine learning application that automatically identifies unwanted SMS messages and demonstrates the practical application of Natural Language Processing and Naive Bayes classification.

## 👩‍💻 Author
Komali Srinuja

