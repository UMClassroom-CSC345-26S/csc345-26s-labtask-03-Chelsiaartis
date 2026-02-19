import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def load_and_normalize_data(filename):
    df = pd.read_csv(filename)
    features = df[['Volume', 'Doors']]
    scaler = MinMaxScaler()
    normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=['Volume', 'Doors'])
    return normalized_features, df['Style']

def export_tree_visualization(dt_model, feature_names, class_names, output_filename):
    plt.figure(figsize=(25, 15)) 
    plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

def main():
    input_file = 'MyCars.csv'
    X, y = load_and_normalize_data(input_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    predictions = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    export_tree_visualization(dt_model, ['Volume', 'Doors'], dt_model.classes_.tolist(), 'TreeCars.png')
    print("SUCCESS: Created 'TreeCars.png'")
    
    results_df = X_test.copy()
    results_df['Style'] = y_test
    results_df['PredictedStyle'] = predictions
    results_df.to_csv('TreeCars.csv', index=False)
    
    with open('TreeCars.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', '', 'Accuracy:', accuracy])
        
    print("SUCCESS: Created 'TreeCars.csv' with accuracy row.")

if __name__ == "__main__":
    main()
