import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize sidebar menu
st.sidebar.title("Menu")
menu_option = st.sidebar.selectbox("Pilih Menu:", ["Sistem Klasifikasi Polisemi", "Train Model"])

# Function to train a custom model
def train_model(data):
    """
    Train a logistic regression model using TF-IDF features.
    """
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['context_1'] + " " + data['context_2'])
    y = data['is_polysemous']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, vectorizer, report

# Train Model Menu
if menu_option == "Train Model":
    st.title("Train a Custom Model")
    st.write("""
    Upload a dataset to train a custom polysemy classification model. 
    The dataset should be a CSV file with the following columns:
    - `word`: The word to analyze.
    - `context_1`: The first context sentence.
    - `context_2`: The second context sentence.
    - `is_polysemous`: A boolean indicating if the word is polysemous.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload dataset (CSV format):", type=["csv"])
    
    if uploaded_file:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        
        # Train the model
        if st.button("Train Model"):
            model, vectorizer, report = train_model(data)
            st.success("Model berhasil ditraining!")
            st.json(report)  # Display evaluation report
            
            # Save the model and vectorizer
            with open("custom_model.pkl", "wb") as model_file:
                pickle.dump((model, vectorizer), model_file)
            st.download_button(
                label="Download Trained Model",
                data=open("custom_model.pkl", "rb").read(),
                file_name="custom_model.pkl",
                mime="application/octet-stream"
            )

# Classify Polysemy Menu
elif menu_option == "Sistem Klasifikasi Polisemi":
    st.title("Sistem Klasifikasi Polisemi")
    st.write("""
    Aplikasi ini mengidentifikasi apakah suatu kata dalam bahasa Indonesia bersifat polisemi (yaitu, memiliki banyak arti dalam konteks yang berbeda). 
    Anda dapat memilih untuk menggunakan model IndoBERT atau mengunggah model khusus.""")
    
    # Model selection
    model_option = st.radio(
        "Pilih Klasifikasi Model:",
        ("Upload Model",)
    )
    
    if model_option == "Upload Model":
        uploaded_model = st.file_uploader("Upload model file yang sudah di training (e.g., .pkl):", type=["pkl"])
        if uploaded_model:
            model, vectorizer = pickle.load(uploaded_model)
            st.success("Model sudah berhasil terpasang!")
        else:
            st.info("Silahkan upload model untuk mulai klasifikasi.")
    
    
    # Input for classification
    word = st.text_input("Masukkan kata untuk dianalisa:")
    context_sentences = st.text_area("Masukkan kalimat (pisah kalimat dengan enter):")
    
    if st.button("Analyze"):
        if not word or not context_sentences:
            st.error("Silahkan isi input teks dan kalimat diatas.")
        else:
            sentences = [sentence.strip() for sentence in context_sentences.split('\n') if sentence.strip()]
            
            if model_option == "IndoBERT":
                prediction = indobert_classifier(" ".join(sentences))[0]
                result = f"'{word}' is classified as {'polysemous' if prediction['label'] == 'LABEL_1' else 'not polysemous'} with confidence {prediction['score']:.2f}."
            elif uploaded_model and model and vectorizer:
                X = vectorizer.transform([" ".join(sentences)])
                is_polysemous = model.predict(X)[0]
                result = f"'{word}' dikategorikan sebagai {'polisemi' if is_polysemous else 'bukan polisemi'}."
            else:
                st.error("No valid model found.")
                result = None
            
            if result:
                st.success(result)
