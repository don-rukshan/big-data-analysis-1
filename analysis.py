import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['mmsi', 'timestamp'], inplace=True)
    return df

def detect_location_anomalies(df, distance_threshold=10):
    anomalies = []
    for mmsi, group in df.groupby('mmsi'):
        group = group.sort_values('timestamp')
        prev_position = None
        for _, row in group.iterrows():
            current_position = (row['latitude'], row['longitude'])
            if prev_position:
                distance = geodesic(prev_position, current_position).kilometers
                if distance > distance_threshold:
                    anomalies.append(row)
            prev_position = current_position
    return pd.DataFrame(anomalies)

def analyze_speed_consistency(df):
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['speed'] = df['distance'] / df['time_diff']
    speed_threshold = df['speed'].quantile(0.99)
    return df[df['speed'] > speed_threshold]

def process_chunk(chunk):
    anomalies = detect_location_anomalies(chunk)
    speed_issues = analyze_speed_consistency(chunk)
    return anomalies, speed_issues

def parallel_process(filepath, num_workers=mp.cpu_count()):
    df = load_and_preprocess(filepath)
    chunks = np.array_split(df, num_workers)
    pool = mp.Pool(num_workers)
    results = pool.map(process_chunk, chunks)
    pool.close()
    pool.join()
    
    anomalies_list, speed_issues_list = zip(*results)
    final_anomalies = pd.concat(anomalies_list)
    final_speed_issues = pd.concat(speed_issues_list)
    
    return final_anomalies, final_speed_issues

def evaluate_performance(df):
    start_time = time.time()
    sequential_results = detect_location_anomalies(df)
    sequential_time = time.time() - start_time
    
    start_time = time.time()
    parallel_results = parallel_process(filepath)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time
    print(f"Sequential Time: {sequential_time} seconds")
    print(f"Parallel Time: {parallel_time} seconds")
    print(f"Speedup: {speedup}")

def plot_anomalies(anomalies):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='longitude', y='latitude', hue='mmsi', data=anomalies)
    plt.title('Detected GPS Spoofing Anomalies')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='MMSI')
    plt.show()

if __name__ == "__main__":
    filepath = "ais_data_2024_09_10.csv"  # Update with actual path
    anomalies, speed_issues = parallel_process(filepath)
    evaluate_performance(pd.read_csv(filepath))
    plot_anomalies(anomalies)
    print(f"Detected Anomalies: {len(anomalies)}")
    print(f"Detected Speed Issues: {len(speed_issues)}")
