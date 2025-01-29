import pandas as pd 
import os 

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import apprendimentoSupervisionato
import warnings
import reteBayesana
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=UserWarning)
## apprendimento supervisionato 
fileName = os.path.join(os.path.dirname(__file__), "./modified/aggregatedDataset.csv")

dataSet = pd.read_csv(fileName)
dataSet = dataSet.drop(columns=['Name'])


# Remove rows where 'Grapes' column has the value 'assente'
dataSet = dataSet[dataSet['Grapes'] != 'assente']

# dataSet.info()
# One-hot encoder for winestyle
wines_enc = pd.get_dummies(dataSet, columns=['WineCategory'])
categorical_cols = [
    col for col in wines_enc.columns if wines_enc[col].dtype == "object"]

label_encoder = LabelEncoder()
for col in categorical_cols:
    wines_enc[col] = label_encoder.fit_transform(wines_enc[col])

differentialColumn = "Rating"

# model = apprendimentoSupervisionato.trainModelKFold(wines_enc,differentialColumn)


discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
continuos_columns = wines_enc.select_dtypes(
    include=['float64', 'int32']).columns
wines_enc[continuos_columns] = discretizer.fit_transform(
    wines_enc[continuos_columns])

# Redefine all float64 variables to int32
wines_enc[continuos_columns] = wines_enc[continuos_columns].astype('int32')

# wines_enc.info()
# print(wines_enc.head())

# print(wines_enc['Rating'].unique())

bayesianNetwork = reteBayesana.bNetCreation(wines_enc)

# # GENERAZIONE DI UN ESEMPIO RANDOMICO e PREDIZIONE DELLA SUA CLASSE
esempioRandom = reteBayesana.generateRandomExample(bayesianNetwork)
print("ESEMPIO RANDOMICO GENERATO\n ", esempioRandom.head())
print(esempioRandom.info())
print("PREDIZIONE DEL SAMPLE RANDOM")
reteBayesana.predici(bayesianNetwork, esempioRandom.to_dict(
    'records')[0], differentialColumn)

# Ipotiziamo un vino senza valutazioni
del (esempioRandom['NumberOfRatings'])
print("ESEMPIO RANDOMICO SENZA Ratings\n", esempioRandom)
print("PREDIZIONE DEL SAMPLE RANDOM SENZA Ratings")
reteBayesana.predici(bayesianNetwork, esempioRandom.to_dict(
    'records')[0], differentialColumn)
