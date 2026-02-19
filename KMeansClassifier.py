import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def load_and_normalize_data(filename):
    df = pd.read_csv(filename)
    features = df[['Volume', 'Doors']]
    scaler = MinMaxScaler()
    normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=['Volume', 'Doors'])
    normalized_features['Style'] = df['Style']
    return normalized_features

def apply_kmeans_clustering(df, num_clusters=5):
    features = df[['Volume', 'Doors']]
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['ClusterID'] = kmeans_model.fit_predict(features)
    cluster_stats = []
    
    for cluster_id in range(num_clusters):
        cars_in_cluster = df[df['ClusterID'] == cluster_id]
        cluster_size = len(cars_in_cluster)
        if cluster_size > 0:
            majority_style = cars_in_cluster['Style'].mode()[0]
            correct_cars = len(cars_in_cluster[cars_in_cluster['Style'] == majority_style])
            accuracy = correct_cars / cluster_size
            cluster_stats.append({
                'ClusterStyle': majority_style,
                'SizeOfCluster': cluster_size,
                'Accuracy': accuracy
            })
            df.loc[df['ClusterID'] == cluster_id, 'ClusterStyle'] = majority_style
            
    accuracy_df = pd.DataFrame(cluster_stats)
    accuracy_df = accuracy_df.groupby('ClusterStyle', as_index=False).agg({
        'SizeOfCluster': 'sum',
        'Accuracy': 'mean' 
    })
    return df, accuracy_df

def main():
    input_file = 'MyCars.csv' 
    df = load_and_normalize_data(input_file)
    clustered_df, accuracy_df = apply_kmeans_clustering(df, num_clusters=5)
    
    clustered_df[['Volume', 'Doors', 'Style', 'ClusterStyle']].to_csv('ClusterCars.csv', index=False)
    print("SUCCESS: Created 'ClusterCars.csv'")
    
    accuracy_df.to_csv('ClusterAccuracy.csv', index=False)
    print("SUCCESS: Created 'ClusterAccuracy.csv'")

if __name__ == "__main__":
    main()
