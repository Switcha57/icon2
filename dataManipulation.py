from urllib.parse import quote_plus
import csv
import pandas as pd
import re
from numpy import nan
red = pd.read_csv("./original_dataset/Red.csv")
white = pd.read_csv("./original_dataset/White.csv")
sparkling = pd.read_csv("./original_dataset/Sparkling.csv")
rose = pd.read_csv("./original_dataset/Rose.csv")


# Naturalmente dato che riguardano gli stessi dati li possiamo unire in unico dataset

red['WineCategory'] = 'red'
white['WineCategory'] = 'white'
sparkling['WineCategory'] = 'sparkling'
rose['WineCategory'] = 'rose'
wines = pd.concat([red, white, sparkling, rose], ignore_index=True)

# Anno è stato rillevato come Object dato la presenza della dicitura NV )non vintage

wines['Year'] = wines['Year'].replace('N.V.', 2025)
wines['Year'] = wines['Year'].astype('int')


def remove_year(name):
    return re.sub(r'\s+\d{4}$', '', name).replace(' N.V.', '')


wines['Name'] = wines['Name'].apply(remove_year)
varieties = pd.read_csv('./modified/Varieties.csv')


wines['Grapes'] = nan

for i, wine in wines.iterrows():
    for variety in varieties['Variety']:
        if variety in wine['Name']:
            wines.at[i, 'Grapes'] = variety
            break
wines.Grapes = wines.Grapes.fillna('assente')

wines.to_csv("./modified/aggregatedDataset.csv", index=False, encoding="utf-8")

# Open the input and output files
with open('./modified/aggregatedDataset.csv', 'r', encoding='utf-8') as infile, \
        open('./modified/aggregatedDatasetSanitazed.csv', 'w', newline='', encoding='utf-8') as outfile:

    # Create a CSV reader and writer
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames  # Get header names
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header row to the output
    writer.writeheader()

    # Process each row
    for row in reader:
        # URL encode all values in the row
        encoded_row = {
            key: quote_plus(value)
            for key, value in row.items()
        }
        writer.writerow(encoded_row)
