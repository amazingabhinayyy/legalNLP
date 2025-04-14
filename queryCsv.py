import pandas as pd

# Load the CSV
df = pd.read_csv('data/preliminary_petitions.csv')

# Print the first element of the 'full_text' column
with open('output.txt', 'w') as f:
    print(df['full_text'].iloc[0], file=f)