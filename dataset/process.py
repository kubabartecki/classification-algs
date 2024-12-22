import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def label_and_normalize():
    df = pd.read_csv('dataset/user_behavior_dataset.csv')
    
    label_encoders = {}
    numeric_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
        else:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
            mapping = {
                str(class_): int(value)
                for class_, value in zip(
                    label_encoders[column].classes_,
                    label_encoders[column].transform(label_encoders[column].classes_)
                )
            }
            print(f"Mapping for '{column}': {mapping}")
    
    columns_to_normalize = [col for col in numeric_columns if col not in ['User ID', 'User Behavior Class']]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    
    df_normalized = pd.DataFrame(normalized_data, columns=columns_to_normalize, index=df.index)
    
    for column in columns_to_normalize:
        df[column] = df_normalized[column]
    
    df.to_csv('dataset/processed_dataset.csv', index=False)
    
    print("Processed Dataset (Head):")
    print(df.head())
