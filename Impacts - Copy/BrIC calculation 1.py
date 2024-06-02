import pandas as pd
import numpy as np
import os

# Directory containing CSV files
directory = 'C:/Users/ameli/OneDrive - Imperial College London/Documents/Schoolwork/Imperial/Masters/Data/Impacts - Copy/ImpactResults/r-processed'


max_values = {}

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Read the CSV file
        excel_file = os.path.join(directory, filename)
        df = pd.read_csv(excel_file)

        # Find the index where la_r is max
        max_la_r_index = df['la_r'].idxmax()
        max_rv_r_index = df['rv_r'].idxmax()
        max_ra_r_index = df['ra_r'].idxmax()

        # Find the corresponding values of la_x, la_y, la_z, rv_x, rv_y, and rv_z
        max_values[filename] = {
            'max_la_x': df.at[max_la_r_index, 'la_x'],
            'max_la_y': df.at[max_la_r_index, 'la_y'],
            'max_la_z': df.at[max_la_r_index, 'la_z'],
            'max_la_r': df.at[max_la_r_index, 'la_r'],
            'max_rv_r': df.at[max_rv_r_index, 'rv_r'],
            'max_ra_r': df.at[max_ra_r_index, 'ra_r'],
            'BRIC' : np.sqrt((df['rv_x'].abs().max()/ 66.25) ** 2 + (df['rv_y'].abs().max() / 56.45) ** 2 +(df['rv_z'].abs().max()/ 42.87) ** 2),
            'rv_x': df['rv_x'].abs().max(),
            'rv_y' : df['rv_y'].abs().max(),
            'rv_z' : df['rv_z'].abs().max(),
            'la_r': np.sqrt(df.at[max_la_r_index, 'la_x']**2 + df.at[max_la_r_index, 'la_y']**2 + df.at[max_la_r_index, 'la_z']**2),
            'rv_r': np.sqrt(df.at[max_rv_r_index, 'rv_x']**2 + df.at[max_rv_r_index, 'rv_y']**2 + df.at[max_rv_r_index, 'rv_z']**2)
        
        }

       

# Create a DataFrame from the dictionary
max_values_df = pd.DataFrame.from_dict(max_values, orient='index')

# Save the DataFrame to an Excel file
output_excel_file = 'max_values_relaxed.xlsx'
max_values_df.to_excel(output_excel_file)

print(f"Maximum values saved to {output_excel_file}")
