from owlready2 import *
import os

rdf_file_path = "mioVino3.rdf"
if os.path.exists(rdf_file_path):
    os.remove(rdf_file_path)
onto = get_ontology("file://mioVino3.rdf")


with onto:
    class Wine(Thing):
        pass
    class Winery(Thing):
        pass
    class Red_wine(Wine):
        pass
    class White_wine(Wine):
        pass
    class Sparkling_wine(Wine):
        pass
    class Rose_wine(Wine):
        pass
    class Grape(Thing):
        pass
with onto:
    class is_from_country(Winery >> str):
        pass
    class is_from_region(Winery >> str):
        pass
    class is_made_by(Wine >> Winery):
        pass
    class is_made_from(Wine >> Grape):
        pass


import pandas as pd

df = pd.read_csv('./modified/aggregatedDataset.csv')

# Seleziona solo le colonne 'region', 'country' e 'winery'cls
df_selected = df[['Region', 'Country', 'Winery']]


for index, row in df_selected.iterrows():
    Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])





onto.save()



