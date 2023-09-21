import streamlit as st
import pandas as pd
import os
from src.utils import load_object


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

# gender, states, df_state_city, occupation_list, employment_type = get_categorical_variables()

def main():

    # Set a background color
    
    html_temp = """
    <div style="background-color: #5CDB95; padding: 20px; text-align: center;">
        <h1 style="color: #EDF5E1; font-size: 26px; font-weight: bold; text-transform: uppercase;">Credit Risk Prediction App</h1>
    </div>
"""

    st.markdown(html_temp,unsafe_allow_html=True)
    # Input fields 
    

    # Custom styling for the subheader
    html_temp_header = """
        <div style="padding: 16px;text-align:left;">
            <h3 style="font-weight: bold; font-size: 20px;">Fill the below Form to Know Customer Default Probability:</h3>
        </div>
        """
    st.markdown(html_temp_header, unsafe_allow_html=True)
    
    
    # Dropdowns
    home_type = ['RENT','OWN','MORTGAGE','OTHER']
    
    # Custom styling for the subheader
    html_temp_header1 = """
        <div style="background-color:#05386B;padding: 10px;text-align:center;">
            <h3 style="color:#EDF5E1;font-weight: bold; font-size: 16px;">DEMOGRAPHIC INFO</h3>
        </div>
        """
    st.markdown(html_temp_header1, unsafe_allow_html=True)

     
    name = st.text_input("Enter Customer Name:","Manoj Gupta")
    
    contact_number = st.text_input("Enter Customer Mobile No:","9876543210")

    age = st.number_input("Enter Customer's Age",18,75,25)
    
    selected_home = st.selectbox(f"Select Home Type:", sorted(home_type), index=0)

    income = st.slider("Select Your Income(in USD)",1000,2000000,5000)
    
    intent_type = ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION']
    
    selected_intent =  st.selectbox(f"Select Loan Purpose:", sorted(intent_type), index=0)
 
    loan_amount = st.number_input("Enter Loan Amount Required",500,300000)
    
    interest_rate = st.number_input("Enter Interest Rate",6.0,30.0,10.0)
    
    credit_history_length = st.slider("Select Your Credit History in Years:",0,30,5)
    
    employment_length = st.slider("Select Your Employment History in Years:",0,40,5)

    # Validation logic
    if len(contact_number)>10:
        st.error("Error: Mobile Number cannot be greater than 10 digits")
        st.stop()

    # Submit button
    if st.button("Check Default Chance"):
        
        data=CustomData(
            age = age,
            income = income,
            credit_history_length = credit_history_length,
            loan_amount = loan_amount,
            interest_rate = interest_rate,
            employment_length = employment_length,
            home_type = selected_home,
            intent_type = selected_intent
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
        st.write(f"Getting Loan Default Chance for, {name}")
        
        if results=='Low':
            st.text("Chances are default are low, you can approve the loan")
        else:
            st.text("Chances of default are high, you can reject the loan")

    # Reset button
    if st.button("Reset"):
        # Clear all inputs and selections
        st.experimental_rerun()

    
if __name__=="__main__":
    main()
