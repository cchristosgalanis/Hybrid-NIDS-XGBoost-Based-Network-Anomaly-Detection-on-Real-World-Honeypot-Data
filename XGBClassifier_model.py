import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import dataset1 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from itertools import cycle
import joblib

def xgbclassifier():
    print("--- XGBoost Multi-Class Classifier ---")

    # loading datasets 
    try:
        result = dataset1.data_preprocess(
            "http_flood (1).csv", 
            "slow_rate_dos.csv",
            "tcp_connect.csv", 
            "tcp_flood.csv", 
            "CIC-IDS- 2017.csv"
        )
        
        if result is None or result[0] is None:
            print("Preprocessing failed.")
            return
            
        x_first, y_first, label_map, scaler = result
        
    except Exception as e:
        print(f"Error calling preprocess: {e}")
        return

    #x_first is already a dataframe
    x_df = x_first.reset_index(drop=True)

    #Label Preparation
    inv_map = {v: k for k, v in label_map.items()}
    y_names = np.array([inv_map.get(val, f"Unknown-{val}") for val in y_first])
    
    # Label Encoder (0, 1, 2...)
    le = LabelEncoder()
    y = le.fit_transform(y_names)
    class_names = le.classes_
    n_classes = len(class_names)
    
    print(f"Multi-Class Detection: Found {n_classes} classes: {class_names}")
    
    # making sure we have two classes
    unique, counts = np.unique(y, return_counts=True)
    rare_classes = unique[counts < 2]
    if len(rare_classes) > 0:
        print(f" Removing rare classes (<2 samples): {rare_classes}")
        mask = ~np.isin(y, rare_classes)
        x_df = x_df[mask]
        y = y[mask]

    # splitting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x_df.values, y, test_size=0.20, random_state=42, stratify=y)

    # Weights
    weights = compute_sample_weight(class_weight='balanced', y=ytrain)

    # Loading model 
    print(f"Starting XGBoost training for {n_classes} classes...")
    model = XGBClassifier(
        objective='multi:softmax', 
        learning_rate=0.1, 
        n_estimators=200, 
        max_depth=6,
        reg_alpha=0.1,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    #model training
    try:
        start_time = time.time()
        model.fit(xtrain, ytrain, sample_weight=weights)
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.2f} seconds")
    except Exception as e:
        print(f"Error while training model: {e}")
        return

    # Predictions
    try:
        start_time = time.time()
        prediction = model.predict(xtest)
        # Για ROC Curve
        y_proba = model.predict_proba(xtest)
        predict_time = time.time() - start_time
    except Exception as e:
        print(f"Error while predicting: {e}")
        return

    # Calculate model's metrics
    accuracy = accuracy_score(ytest, prediction)
    precision = precision_score(ytest, prediction, average='weighted', zero_division=1)
    recall = recall_score(ytest, prediction, average='weighted', zero_division=1)
    f1 = f1_score(ytest, prediction, average='weighted', zero_division=1)

    print(f"\nMetrics: Accuracy {accuracy:.4f} | Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f}")
    print(f"Time: Train {train_time:.4f}s | Predict {predict_time:.4f}s")

    #Classification Report for model
    print("\nClassification Report:")
    print(classification_report(ytest, prediction, target_names=class_names))

    # Plots
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(ytest, prediction)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("Confusion Matrix.png")
    plt.show()

    # Multi-Class ROC Curve
    try:
        plt.figure(figsize=(10, 6))
        y_test_bin = label_binarize(ytest, classes=range(n_classes))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan'])
        
        for i, color in zip(range(n_classes), colors):
            if i in np.unique(ytest):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2,
                         label='ROC {0} (area = {1:0.2f})'.format(class_names[i], roc_auc))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 0.2])
        plt.ylim([0.8, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("ROC Curve.png")
        plt.show()
    except Exception as e:
        print(f"Could not plot ROC: {e}")

    
    # #saving model
    try:
        joblib.dump(model,"XGBClassifier.pkl")
        joblib.dump(scaler,"scaler.pkl")
        print("Model has been saved successfully")
    except RuntimeError:
        print("Error while saving model")


if __name__ == '__main__':
    xgbclassifier()
