import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
vino=pd.read_csv("./modified/aggregatedDataset.csv")


vino.Country.value_counts()

nazioniImportanti = vino.Country.value_counts()[:13] 




# graph = sns.countplot(x='Country', 
#                   data=vino[vino.Country.isin(nazioniImportanti.index.values)],
#                  color='mediumpurple')
# graph.set_title("Nazioni Con quantità di export più elevato", fontsize=20)
# graph.set_xlabel("Nazione", fontsize=15)
# graph.set_ylabel("quantità", fontsize=15)
# graph.set_xticklabels(graph.get_xticklabels(),rotation=45)


graph = sns.countplot(x='Rating', data=vino, color='mediumpurple')
graph.set_title("distribuzione  delle recensioni", fontsize=20)
graph.set_xlabel("Recensione", fontsize=15) 
graph.set_ylabel("Numero", fontsize=15)

# graph1 = sns.histplot(np.log(vino['Price']) , color='#5C0120',kde=True)
# graph1.set_title("Distribuzione Logaritmica del prezzo", fontsize=20) # seting title and size of font

# graph1.set_xlabel("Prezzo(EUR)", fontsize=15) # seting xlabel and size of font
# graph1.set_ylabel("Frequenza", fontsize=15) # seting ylabel and size of font
# graph1.set_xticklabels(np.exp(graph1.get_xticks()).astype(int))

# plt.figure(figsize=(13,5))

# graph = sns.regplot(x=np.log(vino['Price']), y='Rating', 
#                     data=vino, fit_reg=False, color='#5C0120')
# graph.set_title("Rating x Price Distrtibuzione", fontsize=20)
# graph.set_xlabel("Prezzo(EUR)", fontsize= 15)
# graph.set_ylabel("Voto", fontsize= 15)
# graph.set_xticklabels(np.exp(graph.get_xticks()).astype(int))



# plt.show()


# corrs = vino[['Rating','NumberOfRatings','Price','Year']].corr()
# fig, ax = plt.subplots(figsize=(7,5))        

# sns.heatmap(corrs,annot = True,ax=ax,linewidths=.6, cmap = 'YlGnBu');

plt.show()
