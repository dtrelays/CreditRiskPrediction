

# Two Wheeler Loan Approval Scoring App

![2W Loan](artifacts/loan_image.png)

## Overview

The Two Wheeler Loan Approval Scoring App is a data science project that aims to provide a predictive model for assessing the creditworthiness of individuals applying for two-wheeler loans. The project leverages a comprehensive dataset, tailored for the Indian demographic, which includes a wide range of features related to applicants' demographics and financial profiles.

The primary objective of this project is to predict the likelihood of loan approval based on the applicant's profile. The model assigns a score between 0 and 100 to each applicant, with scores above 85 indicating a high chance of loan approval.

This project can help 2W dealers to put in details of prospective customers and get a score on whether he/she is likely to get loan or not.


## Dataset

### Dataset Overview

The dataset used in this project contains a variety of features, including but not limited to:

- Applicant's name
- Age
- Gender
- Employment details
- Income
- Credit history
- Loan amount
- Loan tenure
- LTV ratio
- And more...

These features are used to build a predictive model that assesses the creditworthiness of loan applicants.

### Data Source

- Dataset Source - https://www.kaggle.com/datasets/yashkmd/credit-profile-two-wheeler-loan-dataset 
- The data consists of 15 columns and 278k rows.

## Methodology

### Data Preprocessing

- Data cleaning and handling missing values
- Feature engineering
- One-hot encoding of categorical variables

### Model Building

- Selecting an appropriate machine learning algorithm 
- Training the model on the preprocessed data
- Model evaluation and fine-tuning
- We found that xgboost performed the best 

## Results

- We obtained 0.91 r2 score on test and 0.88 on validation data using Xgboost
- We were able to get feature importance - LTV Ratio, Credit history length,loan tenure and income were among top features

## Usage

You can visit and try on your own -> https://autoloanapproval-6wd28mgszf4gkzkhwbvihe.streamlit.app/

Alternatively you can clone my repository and try on your own in local

```bash
# Clone the repository
git clone https://github.com/dtrelays/AutoLoanApproval.git

# Install dependencies (if any)
pip install -r requirements.txt

# Run the app
streamlit run app.py
