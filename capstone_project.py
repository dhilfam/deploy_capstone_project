# Import Required Libraries
import pandas as pd
import numpy as np
import pickle
import time
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Page Configuration
st.set_page_config(layout="wide", page_title="Capstone Project Fadhil", page_icon=":heart:")
st.sidebar.title("Navigation")
nav = st.sidebar.selectbox("Go to", ("Home", "Dataset", "Exploratory Data Analysis","Modelling", "Prediction", "About"))

# Dataset Page
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)

#Loading images
heartdisease= Image.open('jantungsakit.jpeg')
strongheart =Image.open('jantungsehat.jpeg')

# Function Heart Disease Prediction
def heart():
    st.write("""
        This app predicts the **Heart Disease.**

        Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_manual():
            st.sidebar.header("Manual Input")
            age = st.sidebar.slider("Age", 0, 100, 25)
            cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
            if cp == "Typical Angina":
                cp = 1
            elif cp == "Atypical Angina":
                cp = 2
            elif cp == "Non-anginal Pain":
                cp = 3
            elif cp == "Asymptomatic":
                cp = 4
            thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 200, 100)
            slope = st.sidebar.selectbox("Slope", ("Upsloping", "Flat", "Downsloping"))
            if slope == "Upsloping":
                slope = 1
            elif slope == "Flat":
                slope = 2
            elif slope == "Downsloping":
                slope = 3
            ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
            oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 0.0)
            exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
            if exang == "Yes":
                exang = 1
            elif exang == "No":
                exang = 0
            thal = st.sidebar.selectbox("Thal", ("Normal", "Fixed Defect", "Reversable Defect"))
            if thal == "Normal":
                thal = 1
            elif thal == "Fixed Defect":
                thal = 2
            elif thal == "Reversable Defect":
                thal = 3
            sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
            if sex == "Male":
                sex = 1
            else:
                sex =0
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_manual()

    # Data df
    st.image('jantung.jpeg', width=700)

    if st.sidebar.button("Click Here To Predict"):
        df = input_df.copy()
        st.write(df)
        model = pickle.load(open('best_model_rfs.pkl', 'rb'))
        prediction = model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Heart Disease']
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success('This patient has {}'.format(result[0]))
            if (prediction == 0).any():
                st.image(strongheart)
            else:
                st.image(heartdisease)
            st.balloons()
# Home Page
if nav == "Home":
    st.title("Capstone Project DQLab Fadhil")
    st.write('''
    **Machine Learning Prediction**
    
    Hello! Welcome to [M.Fadhil](https://www.linkedin.com/in/muhammad-fadhil-18aba6259/)'s Machine Learning Dashboard. I'm taking part in the Tetris Batch 4 Program at DQLab Academy. This is my capstone project.
    ''')
    st.image("heart.jpg",
             width=700, caption="Adobe Stock")

    st.write('''
    **Project Overview**
    
    Heart disease refers to a group of conditions that can affect the heart's functioning. 
    This includes coronary artery disease, heart failure, arrhythmias, and many other conditions. 
    Heart disease can be life-threatening and is often associated with risk factors such as high blood pressure, high cholesterol levels, poor dietary habits, smoking, lack of physical activity, and genetic factors.
    Early detection of heart disease needs to be carried out in high risk groups so that they can receive immediate treatment and prevention. This project aims to predict whether someone has heart disease or not based on certain criteria.
    ''')

    st.write('''
    **Project Objective**
    
    The aim of this project is to predict whether someone has heart disease or not based on several specified criteria.
    ''')

elif nav == 'Dataset':
    st.title("Dataset")
    st.write('''
    **Dataset Overview**
    
   The dataset used is the heart disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    This dataset has 303 rows and 14 columns. The `target` column is a column that shows whether someone has heart disease or not. 
    If you have heart disease, then the `target` column value is 1, if you do not have heart disease, then the `target` column value is 0.
    ''')
    st.write('''
    **Dataset Description**
    
    The following is a description of the dataset used.
    
    1. `age` : usia dalam tahun (umur)
    2. `sex` : jenis kelamin (1 = laki-laki; 0 = perempuan)
    3. `cp` : tipe nyeri dada
        - 1: typical angina
        - 2: atypical angina
        - 3: non-anginal pain
        - 4: asymptomatic
    4. `trestbps` : tekanan darah istirahat (dalam mm Hg saat masuk ke rumah sakit)
    5. `chol` : serum kolestoral dalam mg/dl
    6. `fbs` : gula darah puasa > 120 mg/dl (1 = true; 0 = false)
    7. `restecg` : hasil elektrokardiografi istirahat
        - 0: normal
        - 1: memiliki ST-T wave abnormalitas (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        - 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes
    8. `thalach` : detak jantung maksimum yang dicapai
    9. `exang` : angina yang diinduksi oleh olahraga (1 = yes; 0 = no)
    10. `oldpeak` : ST depression yang disebabkan oleh olahraga relatif terhadap istirahat
    11. `slope` : kemiringan segmen ST latihan puncak
        - 1: naik
        - 2: datar
        - 3: turun
    12. `ca` : jumlah pembuluh darah utama (0-3) yang diwarnai dengan flourosopy
    13. `thal` : 3 = normal; 6 = cacat tetap; 7 = cacat yang dapat dibalik
    14. `target` : memiliki penyakit jantung atau tidak (1 = yes; 0 = no)
    ''')

    # show dataset
    st.write('''
    **Show Dataset**
    ''')
    st.dataframe(df.head())

    # show dataset shape
    st.write(f'''**Dataset Shape:** {df.shape}''')

    # show dataset description
    st.write('''
    **Dataset Description**
    ''')
    st.dataframe(df.describe())

    # show dataset count visualization
    st.write('''
    **Dataset Count Visualization**
    ''')
    views = st.selectbox("Select Visualization", ("", "Target", "Age"))
    if views == "Target":
        st.bar_chart(df.target.value_counts())
        st.write('''
        `Target` is a column that shows whether a person has heart disease or not. If you have heart disease, then the `target` column value is 1, if you don't have heart disease, then the `target` column value is 0. 
        Based on the visualization above, it can be seen that the number of people who have heart disease is more than those who don't have heart disease, namely 526 people compared to 499 people.
        ''')
    elif views == "Age":
        st.bar_chart(df['age'].value_counts())
        st.write('''
        Based on the visualization above, it can be seen that the highest number of people who have heart disease are 68 people aged 58 years. 
        Meanwhile, the highest number of people who do not have heart disease is in the range of 74-77 years, as many as 9 people.''')

elif nav == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write('''
    **Data Cleansing**
    
   At this stage, the data is checked to see whether there is empty data or not. If there is empty data, the data will be deleted.
    ''')
    st.write('''
    The information we will dig up is features that have spelling errors::
    1. Feature `CA`: Has 5 values from the range 0-4, therefore the value 4 is changed to NaN (because it shouldn't exist)
    2. Feature `thal`: Has 4 values from the range 0-3, therefore the value 0 is changed to NaN (because it shouldn't exist)
    ''')
    views = st.radio("Show Data", ("CA", "Thal"))
    if views == "CA":
        st.write('''
        **Feature CA**
        
        The CA feature has 5 values from the range 0-4, therefore the value 4 is changed to NaN (because it shouldn't exist)
        ''')
        st.dataframe(df.ca.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.ca.replace(4, np.nan).value_counts().to_frame().transpose())
    elif views == "Thal":
        st.write('''
        **Feature Thal**
        
        The Thal feature has 4 values from the range 0-3, therefore the value 0 is changed to NaN (because it shouldn't exist)
        ''')
        st.dataframe(df.thal.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.thal.replace(0, np.nan).value_counts().to_frame().transpose())

elif nav == "Modelling":
    st.header("Modelling")
    var = st.select_slider("Select Model", ("Before Tuning", "After Tuning", "ROC-AUC", "Conclusion"))
    if var == "Before Tuning":
        accuracy_score = {
            'Logistic Regression': 0.81,
            'Decision Tree': 0.72,
            'Random Forest': 0.81,
            'MLP Classifier': 0.82,
        }
        st.write('''
        **Model Before Tuning**
        
        The following are the accuracy results of the model before tuning.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Based on the accuracy results of the model before tuning, it can be seen that the model with the highest accuracy is the MLP Classifier with an accuracy of 0.82.
        ''')

    elif var == "After Tuning":
        accuracy_score = {
            'Logistic Regression': 0.81,
            'Decision Tree': 0.77,
            'Random Forest': 0.84,
            'MLP Classifier': 0.78,
        }
        st.write('''
        **Model After Tuning**
        
        The following are the accuracy results of the model after tuning.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Based on the accuracy results of the model after tuning, now it can be seen that the model with the highest accuracy is the Random Forest with an accuracy of 0.84.
        ''')

    elif var == "ROC-AUC":
        accuracy_score = {
            'Logistic Regression': 0.88,
            'Decision Tree': 0.79,
            'Random Forest': 0.90,
            'MLP Classifier': 0.83,
        }
        st.write('''
        **Model AUC-ROC Analysis**
        
        The following are the accuracy results of the AUC-ROC Analysis.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Based on the accuracy results of the model AUC-ROC Analysis, now it can be seen that the model with the highest score is the Random Forest with the value of 0.90.
        ''') 
    
        st.image("aucroc.png", width=700)

    elif var == "Conclusion":
        st.write('''
        **Conclusion**
        
        1. So it can be concluded that the model with better performance is the Random Forest model, characterized by the Random Forest model which has the highest accuracy value in the classification report, 
        which is 84% and the Random Forest model has the largest AUC-ROC score compared to other models, which is equal to 90%.

        2. Among the 13 different features studied, it can be identified that 9 main features studied play an important role in differentiating between positive and negative diagnoses. 
        These distinguishing features include the type of chest pain (cp), the maximum heart rate achieved during exercise (thalach), the number of major blood vessels (ca), the degree of ST depression caused by exercise in comparison with the resting state (oldpeak), the slope of the segment ST peak exercise (slope), exercise-induced angina (exang), thallium test results (thal), gender (sex) and age (age).
        ''')

elif nav == 'Prediction':
    st.header("My Apps")
    heart()

elif nav == "About":
    st.title("About Me")
    st.image("Foto Seluruh Badan.jpeg", width=200)
    st.write('''
    **Muhammad Fadhil**
    
    I am a graduate of the Sepuluh Nopember Institute of Technology. I majored in Geomatics Engineering.
    Currently taking part in the Tetris Batch 4 Program at DQLab Academy. This is my Capstone Project.
    ''')
    st.write('''
    **Contact Me**
    
    - [LinkedIn](https://www.linkedin.com/in/muhammad-fadhil-18aba6259/)
    - [Github](https://github.com/dhilfam)
    - [Instagram](https://www.instagram.com/dhilfam/)
    ''')

    select_item = st.selectbox("My Another Project Preview", ('', 'Iris Prediction', 'Housing Price Prediction'))
    if select_item == "Iris Prediction":
        st.header("[Ini Contoh Iris](https://deployiris-mfadhil.streamlit.app/)")
    elif select_item == "Housing Price Prediction":
        st.header("Ini Contoh Housing Price Prediction")
