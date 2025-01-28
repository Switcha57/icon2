import pickle
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()


def visualizeInfo(bayesianNetwork: BayesianNetwork):
    # Ottengo le distribuzioni di probabilità condizionata (CPD)
    cpd_list = bayesianNetwork.get_cpds()
    for cpd in cpd_list:
        print(f"\nCPD per la variabile '{cpd.variable}':")
        print(cpd)
        print("=" * 40)


# Funzione che crea la rete bayesiana
def bNetCreation(dataSet):
    
    edges = []
    edges.append(('Country','Region'))
    edges.append(( 'Region','Winery'))
    edges.append(( 'Region','Grapes'))
    edges.append(('Year', 'Grapes'))
    edges.append(('Winery', 'Grapes'))
    edges.append(('Winery', 'NumberOfRatings'))
    edges.append(('Winery', 'Price'))
    edges.append(('Grapes', 'WineCategory_red'))
    edges.append(('Grapes', 'WineCategory_rose'))
    edges.append(('Grapes', 'WineCategory_sparkling'))
    edges.append(('Grapes', 'WineCategory_white'))
    edges.append(('Grapes', 'Price'))
    edges.append(('WineCategory_red','Price'))

    edges.append(('WineCategory_rose', 'Price'))

    edges.append(('WineCategory_sparkling', 'Price'))

    edges.append(('WineCategory_white', 'Price'))
    edges.append(('NumberOfRatings', 'Rating'))
    edges.append(('Price', 'Rating'))


   

    # Ricerca della struttura ottimale
    hc_k2 = HillClimbSearch(dataSet)
    k2_model = hc_k2.estimate(scoring_method='k2score')
    # Creazione della rete bayesiana

    model = BayesianNetwork(k2_model.edges)
    # model = BayesianNetwork(edges)
    model.fit(dataSet, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    # Salvo la rete bayesiana su file
    with open('modello.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    visualizeInfo(model)
    return model

# Funzione che carica la rete bayesiana da file


def loadBayesianNetwork():
    with open('modello.pkl', 'rb') as input:
        model = pickle.load(input)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model

# Predico il valore di differentialColumn per l'esempio


def predici(bayesianNetwork: BayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example)
    print(result)

# genera un esempio randomico


def generateRandomExample(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['Rating'])
