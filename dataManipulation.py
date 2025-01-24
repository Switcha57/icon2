import pandas as pd

red = pd.read_csv("./original_dataset/Red.csv")
white = pd.read_csv("./original_dataset/White.csv")
sparkling = pd.read_csv("./original_dataset/Sparkling.csv")
rose = pd.read_csv("./original_dataset/Rose.csv")


# Naturalmente dato che riguardano gli stessi dati li possiamo unire in unico dataset

red['WineCategory'] = 'red'
white['WineCategory'] = 'white'
sparkling['WineCategory'] = 'sparkling'
rose['WineCategory'] = 'rose'
wines =  pd.concat([red, white, sparkling, rose], ignore_index=True)

# Anno è stato rillevato come Object dato la presenza della dicitura NV )non vintage

wines['Year'] = wines['Year'].replace('N.V.', 2025) 
wines['Year'] = wines['Year'].astype('int')

wines.to_csv("./modified/aggregatedDataset.csv",index=False,encoding="utf-8")
