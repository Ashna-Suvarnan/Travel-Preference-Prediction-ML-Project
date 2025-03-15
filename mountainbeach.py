import streamlit as st
import pickle
from PIL import Image
import pandas as pd

data = pd.read_csv('Travel preference mountain or beaches.csv')

st.sidebar.title("User Authentication")
user_id = st.sidebar.text_input("User ID")
password = st.sidebar.text_input("Password", type="password")


if user_id == "Ashna Suvarnan" and password == "123456":
    st.sidebar.success("Login successful!")

    def travel():
        
        st.title("Travel Preference Prediction 🌊🏔️")
        image = Image.open('travel image.jpg')
        st.image(image, width=800)

        st.subheader("Select the Features")

        # User input features
        Age = st.selectbox("Age", data['Age'].sort_values().unique())
        Gender = st.selectbox("Gender", data['Gender'].sort_values().unique())
        Income = st.slider("Income ($)", min_value=1000, max_value=1000000, step=1000, value=50000)
        Education_Level = st.selectbox("Education Level", data['Education_Level'].sort_values().unique())
        Travel_Frequency = st.selectbox("Travel Frequency", data['Travel_Frequency'].sort_values().unique())
        Vacation_Budget = st.slider("Vacation Budget ($)", min_value=100, max_value=50000, step=100, value=5000)
        Location = st.selectbox("Location", data['Location'].sort_values().unique())
        Proximity_to_Mountains = st.slider("Proximity to Mountains (km)", min_value=1, max_value=1000, step=1, value=50)
        Proximity_to_Beaches = st.slider("Proximity to Beaches (km)", min_value=1, max_value=1000, step=1, value=50)
        Pets = st.selectbox("Pets", data['Pets'].sort_values().unique())
        Environmental_Concerns = st.selectbox("Environmental Concerns", data['Environmental_Concerns'].sort_values().unique())

        # Create dataframe with user inputs
        df = pd.DataFrame([[Age, Gender, Income, Education_Level, Travel_Frequency, Vacation_Budget, 
                             Location, Proximity_to_Mountains, Proximity_to_Beaches, Pets, Environmental_Concerns]],
                            columns=[
                                'Age', 'Gender', 'Income', 'Education_Level', 'Travel_Frequency', 'Vacation_Budget', 
                                'Location', 'Proximity_to_Mountains', 'Proximity_to_Beaches', 'Pets', 'Environmental_Concerns'
                            ])

        # Load pre-trained model and encoders
        model = pickle.load(open('lr_model.sav', 'rb'))
        scaler = pickle.load(open('sd_scaler.sav', 'rb'))
        ohe1 = pickle.load(open('ohe11_Gender.sav', 'rb'))
        le1 = pickle.load(open('le11_Education.sav', 'rb'))
        ohe2 = pickle.load(open('ohe22_Location.sav', 'rb'))

        # Encode categorical variables
        df['Education_Level'] = le1.transform(df['Education_Level'])

        
        Gender_encoded = ohe1.transform(df[['Gender']])
        dfGender = pd.DataFrame(Gender_encoded, columns=ohe1.get_feature_names_out(['Gender']))

        
        Location_encoded = ohe2.transform(df[['Location']])
        dfLocation = pd.DataFrame(Location_encoded, columns=ohe2.get_feature_names_out(['Location']))

    
        features = pd.concat([df, dfGender, dfLocation], axis=1)
        features.drop(columns=['Gender', 'Location'], inplace=True)

        
        features = features[scaler.feature_names_in_]

        
        pred = st.button('Submit')
        if pred:
            
            feature_scaled = scaler.transform(features)
            
            prediction = model.predict(feature_scaled)

            
            if prediction == 0:
                st.write('Your preference: **Mountains** ⛰️')
            else:
                st.write('Your preference: **Beaches** 🏖️')


    travel()
