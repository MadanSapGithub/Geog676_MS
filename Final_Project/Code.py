import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set the environment variable to prevent memory leak warning (specific to Windows with MKL)
# This is a known issue with KMeans and the Intel Math Kernel Library (MKL), and setting OMP_NUM_THREADS to 1 can prevent the issue.
os.environ["OMP_NUM_THREADS"] = "1"

##################################################### MAY ##############################################################################

# Define the input directory where your CSV file is located
inDir = "C:\\Users\\madan.sapkota\\OneDrive - Texas A&M University\\Desktop\\Model Based"

# Checking if the specified directory exists and is accessible
# This ensures that the directory exists and you can safely load your data from it.
if os.path.isdir(inDir):
    print("Directory exists and is accessible.")  # If the directory exists, this will be printed
else:
    print("Directory does not exist or is not accessible.")  # This will print if the directory is not found

# Load the dataset from the specified directory into a pandas DataFrame
# This assumes the file "Final May F1.csv" exists at the provided location
df1 = pd.read_csv(f"{inDir}\\Final May F1.csv")
print("Loaded DataFrame:")
print(df1.head())  # Displaying the first few rows of the DataFrame to verify the data loaded correctly

# Scale the data - StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1
# Scaling is important because clustering algorithms like K-means are sensitive to the scale of data.
# Without scaling, features with larger ranges could dominate the clustering results.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df1[["NDVI_UAV", "Slope", "COMP", "VWC"]])

# Print the first 5 rows of the scaled features to check the transformed data
print("Scaled Features:")
print(scaled_features[:5])  # This shows the first 5 rows of the scaled data

# Set the random seed for reproducibility (ensures the same results if the code is run multiple times)
np.random.seed(123)

# Perform K-means clustering with 3 clusters (you can adjust the number of clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=123)
df1['cluster'] = kmeans.fit_predict(scaled_features)  # Assign the predicted cluster to a new column in the DataFrame

# Save the DataFrame with the cluster results to a new CSV file in the specified directory
output_path = f"{inDir}\\Cluster_May_F1.csv"
df1.to_csv(output_path, index=False)  # `index=False` ensures the row index is not saved in the CSV file
print(f"Results saved to {output_path}")  # Prints the location of the saved file

# Verify if the file has been saved successfully
# If the file exists at the specified path, it will be loaded and printed for inspection
if os.path.exists(output_path):
    print("File saved successfully.")  # If the file exists, print success message
    # Load the saved CSV file into a new DataFrame to confirm the data has been saved correctly
    saved_df_1 = pd.read_csv(output_path)
    print("Contents of the saved DataFrame:")
    print(saved_df_1)  # Display the first few rows of the saved DataFrame to ensure it contains the clustering results
else:
    print("Error: File was not saved successfully.")  # If the file is not found, print an error message

# Print the centroid values for each cluster. These are the average values of the features for each cluster.
centroid_values = kmeans.cluster_centers_
print("Centroid values for each variable within each cluster:")
print(centroid_values)  # This will show the mean values of the features for each cluster

##################################################### July ##############################################################################

# Load the dataset 
df2 = pd.read_csv(f"{inDir}\\Final July F1.csv")
print("Loaded DataFrame:")
print(df2.head())  # Displaying the first few rows of the DataFrame to verify the data loaded correctly

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df2[["NDVI_UAV", "Slope", "COMP", "VWC"]])

# Print the first 5 rows of the scaled features 
print("Scaled Features:")
print(scaled_features[:5])  

# Set the random seed 
np.random.seed(123)

# Perform K-means clustering with 3 CLUSTERS
kmeans = KMeans(n_clusters=3, random_state=123)
df2['cluster'] = kmeans.fit_predict(scaled_features)  # Assign the predicted cluster to a new column in the DataFrame

# Save the DataFrame 
output_path = f"{inDir}\\Cluster_July_F1.csv"
df2.to_csv(output_path, index=False)  # `index=False` ensures the row index is not saved in the CSV file
print(f"Results saved to {output_path}")  # Prints the location of the saved file

# Verify if the file has been saved successfully
if os.path.exists(output_path):
    print("File saved successfully.")  # If the file exists, print success message
    # Load the saved CSV file into a new DataFrame to confirm the data has been saved correctly
    saved_df_2 = pd.read_csv(output_path)
    print("Contents of the saved DataFrame:")
    print(saved_df_2)  # Display the first few rows of the saved DataFrame to ensure it contains the clustering results
else:
    print("Error: File was not saved successfully.")  # If the file is not found, print an error message

# Print the centroid values for each cluster. These are the average values of the features for each cluster.
centroid_values = kmeans.cluster_centers_
print("Centroid values for each variable within each cluster:")
print(centroid_values)  # This will show the mean values of the features for each cluster

##################################################### August ##############################################################################

# Load the dataset from the specified directory into a pandas DataFrame
df3 = pd.read_csv(f"{inDir}\\Final August F1.csv")
print("Loaded DataFrame:")
print(df3.head())  # Displaying the first few rows of the DataFrame to verify the data loaded correctly

# Scale the data - StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1
# Scaling is important because clustering algorithms like K-means are sensitive to the scale of data.
# Without scaling, features with larger ranges could dominate the clustering results.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df3[["NDVI_UAV", "Slope", "COMP", "VWC"]])

# Print the first 5 rows of the scaled features to check the transformed data
print("Scaled Features:")
print(scaled_features[:5])  # This shows the first 5 rows of the scaled data

# Set the random seed for reproducibility (ensures the same results if the code is run multiple times)
np.random.seed(123)

# Perform K-means clustering with 3 clusters (you can adjust the number of clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=123)
df3['cluster'] = kmeans.fit_predict(scaled_features)  # Assign the predicted cluster to a new column in the DataFrame

# Save the DataFrame with the cluster results to a new CSV file in the specified directory
output_path = f"{inDir}\\Cluster_August_F1.csv"
df3.to_csv(output_path, index=False)  # `index=False` ensures the row index is not saved in the CSV file
print(f"Results saved to {output_path}")  # Prints the location of the saved file

# Verify if the file has been saved successfully
# If the file exists at the specified path, it will be loaded and printed for inspection
if os.path.exists(output_path):
    print("File saved successfully.")  # If the file exists, print success message
    # Load the saved CSV file into a new DataFrame to confirm the data has been saved correctly
    saved_df_3 = pd.read_csv(output_path)
    print("Contents of the saved DataFrame:")
    print(saved_df_3)  # Display the first few rows of the saved DataFrame to ensure it contains the clustering results
else:
    print("Error: File was not saved successfully.")  # If the file is not found, print an error message

# Print the centroid values for each cluster. These are the average values of the features for each cluster.
centroid_values = kmeans.cluster_centers_
print("Centroid values for each variable within each cluster:")
print(centroid_values)  # This will show the mean values of the features for each cluster

##################################################### October ##############################################################################

# Load the dataset from the specified directory into a pandas DataFrame
df4 = pd.read_csv(f"{inDir}\\Final October F1.csv")
print("Loaded DataFrame:")
print(df4.head())  # Displaying the first few rows of the DataFrame to verify the data loaded correctly

# Scale the data - StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1
# Scaling is important because clustering algorithms like K-means are sensitive to the scale of data.
# Without scaling, features with larger ranges could dominate the clustering results.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df4[["NDVI_UAV", "Slope", "COMP", "VWC"]])

# Print the first 5 rows of the scaled features to check the transformed data
print("Scaled Features:")
print(scaled_features[:5])  # This shows the first 5 rows of the scaled data

# Set the random seed for reproducibility (ensures the same results if the code is run multiple times)
np.random.seed(123)

# Perform K-means clustering with 3 clusters (you can adjust the number of clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=123)
df4['cluster'] = kmeans.fit_predict(scaled_features)  # Assign the predicted cluster to a new column in the DataFrame

# Save the DataFrame with the cluster results to a new CSV file in the specified directory
output_path = f"{inDir}\\Cluster_October_F1.csv"
df4.to_csv(output_path, index=False)  # `index=False` ensures the row index is not saved in the CSV file
print(f"Results saved to {output_path}")  # Prints the location of the saved file

# Verify if the file has been saved successfully
# If the file exists at the specified path, it will be loaded and printed for inspection
if os.path.exists(output_path):
    print("File saved successfully.")  # If the file exists, print success message
    # Load the saved CSV file into a new DataFrame to confirm the data has been saved correctly
    saved_df_4 = pd.read_csv(output_path)
    print("Contents of the saved DataFrame:")
    print(saved_df_4)  # Display the first few rows of the saved DataFrame to ensure it contains the clustering results
else:
    print("Error: File was not saved successfully.")  # If the file is not found, print an error message

# Print the centroid values for each cluster. These are the average values of the features for each cluster.
centroid_values = kmeans.cluster_centers_
print("Centroid values for each variable within each cluster:")
print(centroid_values)  # This will show the mean values of the features for each cluster



