import streamlit as st
import pandas as pd
import os
from src.utils import load_object,get_categorical_variables


from src.pipeline.predict_pipeline import CustomData,PredictPipeline

@st.cache_resource
def load_model():

    model_path=os.path.join("artifacts","model.pkl")
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

    model=load_object(file_path=model_path)

    preprocessor=load_object(file_path=preprocessor_path)

    loaded_model = model["model"]
    loaded_params = model["params"]

    # Create a new instance of the model with the loaded hyperparameters
    model_with_loaded_params = loaded_model.set_params(**loaded_params)            

    return model_with_loaded_params,preprocessor


model,preprocessor=load_model()

gender, states, df_state_city, occupation_list, employment_type = get_categorical_variables()

def main():

    # Set a background color
    
    html_temp = """
    <div style="background-color: #5CDB95; padding: 20px; text-align: center;">
        <h1 style="color: #EDF5E1; font-size: 26px; font-weight: bold; text-transform: uppercase;">Two Wheeler Loan Approval Scoring App</h1>
    </div>
"""

    st.markdown(html_temp,unsafe_allow_html=True)
    # Input fields 
    
    

    # Custom styling for the subheader
    html_temp_header = """
        <div style="padding: 16px;text-align:left;">
            <h3 style="font-weight: bold; font-size: 20px;">Fill the below Form to Know your Two Wheeler Loan Score</h3>
        </div>
        """
    st.markdown(html_temp_header, unsafe_allow_html=True)
    
    
    # Dropdowns
    dropdown_states = states
    
    # Custom styling for the subheader
    html_temp_header1 = """
        <div style="background-color:#05386B;padding: 10px;text-align:center;">
            <h3 style="color:#EDF5E1;font-weight: bold; font-size: 16px;">DEMOGRAPHIC INFO</h3>
        </div>
        """
    st.markdown(html_temp_header1, unsafe_allow_html=True)

     
    name = st.text_input("Enter Your Name:","Rohan Sharma")
    
    contact_number = st.text_input("Enter Your Mobile No:","9876543210")
        
    selected_gender =  st.selectbox(f"Select Gender:", sorted(gender), index=0)

    age = st.number_input("Enter Your Age",18,75,25)
    
    selected_state = st.selectbox(f"Select State:", sorted(dropdown_states), index=0)

    city_list = df_state_city[df_state_city['State'] == selected_state]['City'].unique().tolist()

    selected_city =  st.selectbox(f"Select City:", sorted(city_list), index=0)
 
    income = st.slider("Select Your Income",15000,300000,50000)
        
    
    html_temp_header2 = """
        <div style="background-color: #05386B;padding: 10px;text-align:center;">
            <h3 style="color: #EDF5E1;font-weight: bold; font-size: 16px;">VEHICLE INFO</h3>
        </div>
        """
    st.markdown(html_temp_header2, unsafe_allow_html=True)

    
    bike_price = st.number_input("Enter Two Wheeler On Road Price",0,350000,50000)
    loan_amount = st.number_input("Enter Loan Amount Required",20000,300000)
    loan_tenure = st.slider("Select Loan Tenure in Years:",1,8)
    

    html_temp_header3 = """
    <div style="background-color: #05386B;padding: 10px;text-align:center;">
        <h3 style="color: #EDF5E1;font-weight: bold; font-size: 16px;">BANKING INFO</h3>
    </div>
    """

    st.markdown(html_temp_header3, unsafe_allow_html=True)
    
    no_of_existing_loans = st.slider("Select No of Existing Loans:",0,10,2)
    
    credit_score = st.slider("Select Your Credit Score:",300,850)
    
    credit_history_length = st.slider("Select Your Credit History in Years:",0,30,5)
    
    existing_customer =  st.selectbox(f"Are You an Existing Bank Customer?", ['Yes','No'], index=0)


    html_temp_header4 = """
    <div style="background-color: #05386B;padding: 10px;text-align:center;">
        <h3 style="color: #EDF5E1;font-weight: bold; font-size: 16px;">EMPLOYMENT INFO</h3>
    </div>
    """

    st.markdown(html_temp_header4, unsafe_allow_html=True)

    selected_employment_type =  st.selectbox(f"Select Your Employment Type:", sorted(employment_type), index=0)

    selected_occupation =  st.selectbox(f"Select Your Occupation:", sorted(occupation_list), index=0)

    # Validation logic
    if loan_amount > bike_price:
        st.error("Error: Loan Amount cannot be greater than Bike Price")
        st.stop()
        
    if len(contact_number)>10:
        st.error("Error: Mobile Number cannot be greater than 10 digits")
        st.stop()

    # Submit button
    if st.button("Get Score"):
        
        data=CustomData(
            age = age,
            gender = selected_gender,
            income = income,
            credit_score = credit_score,
            credit_history_length = credit_history_length,
            no_of_existing_loans = no_of_existing_loans,
            loan_amount = loan_amount,
            loan_tenure = loan_tenure,
            existing_customer = existing_customer,
            state = selected_state,
            city = selected_city,
            employment_type = selected_employment_type,
            occupation = selected_occupation,
            bike_price = bike_price
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df,model,preprocessor)
        
        #line 104 is giving error, 

        print("Output from Model")        
        print(results)

        # Display the score
        st.write(f"Getting Your Profile Score, {name}")
        st.write(f"Your Score: {round(results[0],0)}")
        st.text("If the score is above 85, high chance of loan approval")
        
        st.text("We got your details, our team will reach out to you shortly if you are eligible")


    # Reset button
    if st.button("Reset"):
        # Clear all inputs and selections
        st.experimental_rerun()

    
if __name__=="__main__":
    main()
