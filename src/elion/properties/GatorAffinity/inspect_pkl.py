import pickle
import pandas as pd

file_path = 'LP-PDBbind/test.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"--- File Metadata ---")
print(f"Type of data: {type(data)}")
print(f"Number of items: {len(data)}")

# Print the structure of the first item to see keys and value types
if len(data) > 0:
    print(f"\n--- Structure of First Entry ---")
    first_item = data[0]
    for key, value in first_item.items():
        print(f"Key: {key:<12} | Type: {type(value).__name__:<10} | Value: {value}")

# Optional: Print as a table for better readability using Pandas
print(f"\n--- First 5 Entries (DataFrame view) ---")
df = pd.DataFrame(data[:5])
print(df)