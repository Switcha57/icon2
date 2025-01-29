from owlready2 import *
import os

rdf_file_path = "mioVinoIndividui.rdf"
if os.path.exists(rdf_file_path):
    os.remove(rdf_file_path)
onto = get_ontology("file://mioVinoIndividui.rdf")

# Carico le classi nel ontologia
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

    class Rating(Thing):
        pass

# Carico le proprietà e relazione nel ontologia

with onto:
    class is_from_country(Winery >> str):
        pass

    class is_from_region(Winery >> str):
        pass

    class is_made_by(Wine >> Winery):
        pass

    class has_produced(Winery >> Wine):
        inverse_property = is_made_by

    class is_made_from(Wine >> Grape):
        pass

    class has_been_used_in(Grape >> Wine):
        inverse_property = is_made_from

    class made_in(Wine >> str):
        pass

    class number_of_rating(Rating >> int):
        pass

    class average_rating(Rating >> float):
        pass

    class is_review(Rating >> Wine):
        pass

    class has_been_reviewed(Wine >> Rating):
        pass


# carica gli individui nel ontologia
import pandas as pd

df = pd.read_csv('./modified/aggregatedDatasetSanitazed.csv')

df_selected = df[['Region', 'Country', 'Winery']]


for index, row in df_selected.iterrows():
    Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])


for index,row in df.iterrows():
    if row['WineCategory'] == "red":
        Red_wine(row['Name'],namespace=onto,is_made_by=[Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])],is_made_from=[Grape(row['Grapes'])])
    elif row['WineCategory'] == "rose":
        Rose_wine(row['Name'],namespace=onto,is_made_by=[Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])],is_made_from=[Grape(row['Grapes'])])
    elif row['WineCategory'] == "white":
        White_wine(row['Name'],namespace=onto,is_made_by=[Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])],is_made_from=[Grape(row['Grapes'])])
    elif row['WineCategory'] == "sparkling":
        Sparkling_wine(row['Name'],namespace=onto,is_made_by=[Winery(row['Winery'],namespace=onto,is_from_country=[row['Country']],is_from_region=[row['Region']])],is_made_from=[Grape(row['Grapes'])])
    else :
        print("Riga malformata")


onto.save()
