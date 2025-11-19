import pandas as pd
import os 
import functions 
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import joblib

def data_preprocess(filename1, filename2, filename3, filename4, filename5):
    # target column
    target_column = 'Label'
    columns_to_drop = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port'] 

    files_config = [
        {"path": filename1, "label": "HTTP_FLOOD", "type": "force"},
        {"path": filename2, "label": "SLOW_RATE_DOS", "type": "force"},
        {"path": filename3, "label": "TCP_CONNECT", "type": "force"},
        {"path": filename4, "label": "TCP_FLOOD", "type": "force"},
        {"path": filename5, "label": "mixed", "type": "keep"}
    ]

    dataframes = []

    print("--- Loading Datasets ---")
    for f in files_config:
        try:
            print(f"Loading {f['path']}...", end=" ")
            df_temp = pd.read_csv(f['path'])
            df_temp.columns = df_temp.columns.str.strip()
            
            if f['type'] == "force":
                df_temp[target_column] = f['label']
            elif f['type'] == "keep":
                if target_column in df_temp.columns:
                    df_temp[target_column] = df_temp[target_column].astype(str).str.strip()
                else:
                    print(f"\nCRITICAL ERROR: No Label column in {f['path']}")
                    continue

            dataframes.append(df_temp)
            print(f"Done. Shape: {df_temp.shape}")

        except FileNotFoundError:
            print(f"\nError: File {f['path']} not found.")
            return None, None, None, None

    # concat datasets to one
    try:
        df = pd.concat(dataframes, axis=0, ignore_index=True)
        print(f"All datasets combined. Total shape: {df.shape}")
        print(f"Unique Labels: {df[target_column].unique()}")
    except Exception as e:
        print(f"Error combining datasets: {e}")
        return None, None, None, None

    print("Cleaning features...")
    
    #  Drop single value cols
    single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if single_value_cols:
        df.drop(columns=single_value_cols, inplace=True)
    
    #  Drop Correlated Features (
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Dropped {len(to_drop)} correlated columns.")
    
    # Label Encoding 
    try:
        df, label_map = functions.label_encode_column(df, target_column)
        y_encoded = df[target_column].values 
        df = df.drop(columns=[target_column]) 
        print(f"Labels encoded. Map: {label_map}")
    except Exception as e:
        print(f"Error encoding labels: {e}")
        return None, None, None, None

    # Drop useless columns
    try:
        existing_cols = [c for c in columns_to_drop if c in df.columns]
        if existing_cols:
            df = functions.drop_train_unused(df, existing_cols)
    except:
        pass
    
    # Missing Values
    X_df = functions.fill_missing_values(df)

    # Categorical Handling
    cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        X_numerical = X_df.drop(columns=cat_cols, errors='ignore')
    else:
        X_numerical = X_df
        
    functions.inf_correction(X_numerical)

    # Scaling
    print("Applying RobustScaler...")
    scaler = RobustScaler()
    X_scaled_array = scaler.fit_transform(X_numerical)
    X_final = pd.DataFrame(X_scaled_array, columns=X_numerical.columns)

    # saving label map in order to decode model's result 
    try:
        joblib.dump(label_map,"label_decode_map.pkl")
        print("Label decode map has been saved")
    except Exception as e:
        print(f"Error while saving decode map: {e}")


    return X_final, y_encoded, label_map, scaler