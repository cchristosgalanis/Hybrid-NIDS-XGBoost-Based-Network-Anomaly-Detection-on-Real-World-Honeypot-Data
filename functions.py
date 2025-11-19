import pandas as pd
import numpy as np

#Function to remove rows with NaN values
def drop_nan(df): #take the dataframe as argument 
    try:
        if isinstance(df,pd.DataFrame):
            df.dropna(inplace=True) #
            print("Nan collumns have been removed!")
        else:
            print("There was not NaN collumns")
    except FileNotFoundError:
        print("File was not found")
    #return dataframe
    return df

#function to categorize protocols
def categorize_protocol(df,collumn_name,start_num = 1):
  codes, unique_protocols = df[collumn_name].factorize() #using factorize(), in order to categorize datas in each column
  encoded_protocols = codes + start_num
  protocol_map = {protocol: index+start_num for index,
                  protocol in enumerate(unique_protocols)} #creating dict in order to have knowledge for every categorized protocol
  
  df[collumn_name] = encoded_protocols 
  #return dataframe
  return df

#Function to fill missing values(Imputation)
def fill_missing_values(df):
    if not isinstance(df,pd.DataFrame):
        print("Error: Input is not a Pandas DataFrame")
        return df
    
    #Fill NaN values in numerical columns with the median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    #Fill NaN values in cateforical columns with the mode(most frequent value)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print('NaN values have been filling using median(numerical) and mode (categorical)')
    return df

#Function to drop columns that are not needed for training
def drop_train_unused(df,collumn_name):
    df.drop(collumn_name,axis=1,inplace=True,errors='ignore')
    print(f"Columns {collumn_name} have been dropped")
    #return dataframe
    return df

#function which categorize each label (e.g attack or normal)
def encode_label(df,collumn_name,start_num=1):
    codes,unique_protocols = df[collumn_name].factorize() #similar build with categorize_protocol
    encode_labels = codes + start_num
    label_map = {label: index+start_num for index,
                 label in enumerate(unique_protocols)} 
    
    df[collumn_name] = encode_labels
    print(f"Target column {collumn_name} has been Label Encoded")
    #retrun dataframe
    return df,label_map

#Function to Label Encode a column(e.g protocols or services)
def label_encode_column(df,column_name,start_num=1):
    #factorize() is used to categorize the data in the specific column
    codes, unique_protocols = df[column_name].factorize()
    encode_protocols = codes + start_num
    #Create the dictionary to map codes back to original protocols
    protocol_map = {protocol: index+start_num for index,
                   protocol in enumerate(unique_protocols)}
    
    df[column_name] = encode_protocols #the change is applied directly to the column
    print(f"Column {column_name} has been Label Encoded")

    return df, protocol_map

#function for change inf values
def inf_correction(x):
    x.replace([np.inf,-np.inf],np.nan,inplace=True)
    for column in x.columns:
        if x[column].isnull().any():
            median_val = x[column].median()
            x[column].fillna(median_val,inplace=True)
    


















   

    



    
    
    

    



    



