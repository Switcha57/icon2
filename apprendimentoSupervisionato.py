from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, learning_curve, train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import json
from sklearn.linear_model import Ridge  # ||y - Xw||^2_2 + alpha * ||w||^2_2#
from sklearn.dummy import DummyRegressor
# Funzione che mostra la curva di apprendimento per ogni modello


def plot_learning_curves(model, X, y, differentialColumn, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=10, scoring='neg_root_mean_squared_error')

    # Calcola gli errori su addestramento e test
    train_errors = train_scores
    test_errors = test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    log_message = (f"{model_name} - Train Error Std: {train_errors_std[-1]}, "
                   f"Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, "
                   f"Test Error Var: {test_errors_var[-1]}")
    # Log the message to a file
    with open('learning_curve_log.txt', 'a') as log_file:
        log_file.write(log_message + '\n')

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = np.mean(train_scores, axis=1) * -1
    mean_test_errors = np.mean(test_scores, axis=1) * -1

    # Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors,
             label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors,
             label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()


# Funzione che restituisce i migliori iperparametri per ogni modello
def returnBestHyperparametres(dataset, differentialColumn):
    X = dataset.drop(differentialColumn, axis=1).to_numpy()
    y = dataset[differentialColumn].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ridgeReg = Ridge()
    catboost = CatBoostRegressor()
    RidgeRegressorHyperparameters = {
        'Ridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 14, 15, 16],
        'Ridge__solver': ['auto'],
    }

    CatBoostHyperparameters = {
        'CatBoost__iterations': [100, 200, 300],
        'CatBoost__depth': [ 6, 7, 8],
        'CatBoost__learning_rate': [0.01, 0.05, 0.1],
        'CatBoost__l2_leaf_reg': [1, 3],
        'CatBoost__verbose': [False]
    }

    gridSearchCV_ridgeReg = GridSearchCV(
        Pipeline([('Ridge', ridgeReg)]), RidgeRegressorHyperparameters, cv=5)
    gridSearchCV_catboost = GridSearchCV(
        Pipeline([('CatBoost', catboost)]), CatBoostHyperparameters, cv=5)
    # gridSearchCV_linearReg = GridSearchCV(
    #     Pipeline([('Linear', linearReg)]), LinearRegressorHyperparameters, cv=5)
    # gridSearchCV_dtc = GridSearchCV(
    #     Pipeline([('DecisionTree', dtc)]), DecisionTreeHyperparameters, cv=5)
    # gridSearchCV_rfc = GridSearchCV(
    #     Pipeline([('RandomForest', rfc)]), RandomForestHyperparameters, cv=5)
    # gridSearchCV_reg = GridSearchCV(Pipeline(
    #     [('LogisticRegression', reg)]), LogisticRegressionHyperparameters, cv=5)

    gridSearchCV_ridgeReg.fit(X_train, y_train)
    gridSearchCV_catboost.fit(X_train, y_train)
    # gridSearchCV_linearReg.fit(X_train, y_train)
    # gridSearchCV_dtc.fit(X_train, y_train)
    # gridSearchCV_rfc.fit(X_train, y_train)
    # gridSearchCV_reg.fit(X_train, y_train)
    bestParameters = {
        'Ridge__alpha': gridSearchCV_ridgeReg.best_params_['Ridge__alpha'],
        'Ridge__solver': gridSearchCV_ridgeReg.best_params_['Ridge__solver'],
        'CatBoost__iterations': gridSearchCV_catboost.best_params_['CatBoost__iterations'],
        'CatBoost__depth': gridSearchCV_catboost.best_params_['CatBoost__depth'],
        'CatBoost__learning_rate': gridSearchCV_catboost.best_params_['CatBoost__learning_rate'],
        'CatBoost__l2_leaf_reg': gridSearchCV_catboost.best_params_['CatBoost__l2_leaf_reg'],
        # 'Linear__fit_intercept': gridSearchCV_linearReg.best_params_['Linear__fit_intercept'],
       
        # 'DecisionTree__criterion': gridSearchCV_dtc.best_params_['DecisionTree__criterion'],
        # 'DecisionTree__max_depth': gridSearchCV_dtc.best_params_['DecisionTree__max_depth'],
        # 'DecisionTree__min_samples_split': gridSearchCV_dtc.best_params_['DecisionTree__min_samples_split'],
        # 'DecisionTree__min_samples_leaf': gridSearchCV_dtc.best_params_['DecisionTree__min_samples_leaf'],
        # 'RandomForest__n_estimators': gridSearchCV_rfc.best_params_['RandomForest__n_estimators'],
        # 'RandomForest__max_depth': gridSearchCV_rfc.best_params_['RandomForest__max_depth'],
        # 'RandomForest__min_samples_split': gridSearchCV_rfc.best_params_['RandomForest__min_samples_split'],
        # 'RandomForest__min_samples_leaf': gridSearchCV_rfc.best_params_['RandomForest__min_samples_leaf'],
        # 'RandomForest__criterion': gridSearchCV_rfc.best_params_['RandomForest__criterion'],
        # 'LogisticRegression__C': gridSearchCV_reg.best_params_['LogisticRegression__C'],
        # 'LogisticRegression__penalty': gridSearchCV_reg.best_params_['LogisticRegression__penalty'],
        # 'LogisticRegression__solver': gridSearchCV_reg.best_params_['LogisticRegression__solver'],
        # 'LogisticRegression__max_iter': gridSearchCV_reg.best_params_['LogisticRegression__max_iter']
    }
    return bestParameters


# Funzione che esegue il training del modello mediante cross validation
def trainModelKFold(dataSet, differentialColumn):
    model = {
        'Ridge': {
            'neg_root_mean_squared_error': [],
            'r2': [],
        },
        'CatBoost': {
            'neg_root_mean_squared_error': [],
            'r2': [],
        },
        'Dummy': {
            'neg_root_mean_squared_error': [],
            'r2': [],
        }
        # 'Linear': {
        #     'neg_root_mean_squared_error': [],
        #     'r2': [],
        # }
    }
    bestParameters = returnBestHyperparametres(dataSet, differentialColumn)

    # Log best parameters to a file
    with open('best_parameters.json', 'w') as file:
        json.dump(bestParameters, file, indent=4)
    X = dataSet.drop(differentialColumn, axis=1).to_numpy()
    y = dataSet[differentialColumn].to_numpy()
    # dtc = DecisionTreeClassifier(criterion=bestParameters['DecisionTree__criterion'],
    #                              splitter='best',
    #                              max_depth=bestParameters['DecisionTree__max_depth'],
    #                              min_samples_split=bestParameters['DecisionTree__min_samples_split'],
    #                              min_samples_leaf=bestParameters['DecisionTree__min_samples_leaf'])
    # rfc = RandomForestClassifier(n_estimators=bestParameters['RandomForest__n_estimators'],
    #                              max_depth=bestParameters['RandomForest__max_depth'],
    #                              min_samples_split=bestParameters['RandomForest__min_samples_split'],
    #                              min_samples_leaf=bestParameters['RandomForest__min_samples_leaf'],
    #                              criterion=bestParameters['RandomForest__criterion'])
    # reg = LogisticRegression(C=bestParameters['LogisticRegression__C'],
    #                          penalty=bestParameters['LogisticRegression__penalty'],
    #                          solver=bestParameters['LogisticRegression__solver'],
    #                          max_iter=bestParameters['LogisticRegression__max_iter'])
    ridge = Ridge(alpha=bestParameters['Ridge__alpha'],
                  solver=bestParameters['Ridge__solver'])
    catboost = CatBoostRegressor(iterations=bestParameters['CatBoost__iterations'],
                                 depth=bestParameters['CatBoost__depth'],
                                 learning_rate=bestParameters['CatBoost__learning_rate'],
                                 l2_leaf_reg=bestParameters['CatBoost__l2_leaf_reg'])
    dummy = DummyRegressor(strategy="mean")
    # linear = LinearRegression(fit_intercept=bestParameters['Linear__fit_intercept'])
    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    scoring_metrics = ['neg_root_mean_squared_error', 'r2']

    results_ridge = {}
    results_catboost = {}
    results_dummy = {}
    # results_linear = {}
    for metric in scoring_metrics:
        # scores_dtc = cross_val_score(dtc, X, y, scoring=metric, cv=cv)
        # scores_rfc = cross_val_score(rfc, X, y, scoring=metric, cv=cv)
        # scores_reg = cross_val_score(reg, X, y, scoring=metric, cv=cv)
        score_ridge = cross_val_score(ridge, X, y, scoring=metric, cv=cv)
        score_catboost = cross_val_score(catboost, X, y, scoring=metric, cv=cv)
        score_dummy = cross_val_score(dummy, X, y, scoring=metric, cv=cv)
        # score_linear = cross_val_score(linear, X, y, scoring=metric, cv=cv)
        # results_dtc[metric] = scores_dtc
        # results_rfc[metric] = scores_rfc
        # results_reg[metric] = scores_reg
        results_ridge[metric] = score_ridge
        results_catboost[metric] = score_catboost
        results_dummy[metric] = score_dummy
        # results_linear[metric] = score_linear
    # model['LogisticRegression']['accuracy_list'] = (results_reg['accuracy'])
    # model['LogisticRegression']['precision_list'] = (
    #     results_reg['precision_macro'])
    # model['LogisticRegression']['recall_list'] = (results_reg['recall_macro'])
    # model['LogisticRegression']['f1'] = (results_reg['f1_macro'])
    # model['DecisionTree']['accuracy_list'] = (results_dtc['accuracy'])
    # model['DecisionTree']['precision_list'] = (results_dtc['precision_macro'])
    # model['DecisionTree']['recall_list'] = (results_dtc['recall_macro'])
    # model['DecisionTree']['f1'] = (results_dtc['f1_macro'])
    # model['RandomForest']['accuracy_list'] = (results_rfc['accuracy'])
    # model['RandomForest']['precision_list'] = (results_rfc['precision_macro'])
    # model['RandomForest']['recall_list'] = (results_rfc['recall_macro'])
    # model['RandomForest']['f1'] = (results_rfc['f1_macro'])
    model['Ridge']['neg_root_mean_squared_error'] = (
        results_ridge['neg_root_mean_squared_error'])
    model['Ridge']['r2'] = (results_ridge['r2'])
    model['CatBoost']['neg_root_mean_squared_error'] = (
        results_catboost['neg_root_mean_squared_error'])
    model['CatBoost']['r2'] = (results_catboost['r2'])
    model['Dummy']['neg_root_mean_squared_error'] = (
        results_dummy['neg_root_mean_squared_error'])
    model['Dummy']['r2'] = (results_dummy['r2'])
    # model['Linear']['neg_root_mean_squared_error'] = (
    #     results_linear['neg_root_mean_squared_error'])
    # model['Linear']['r2'] = (results_linear['r2'])

    plot_learning_curves(ridge, X, y, differentialColumn, 'Ridge')
    plot_learning_curves(catboost, X, y, differentialColumn, 'CatBoost')
    plot_learning_curves(dummy, X, y, differentialColumn, 'Dummy')
    # plot_learning_curves(linear, X, y, differentialColumn, 'Linear')
    # plot_learning_curves(rfc, X, y, differentialColumn, 'RandomForest')
    # plot_learning_curves(reg, X, y, differentialColumn, 'LogisticRegression')
    visualizeMetricsGraphs(model)
    return model

# Funzione che visualizza i grafici delle metriche per ogni modello


def visualizeMetricsGraphs(model):
    models = list(model.keys())
    
    # Creazione di un array numpy per ogni metrica
    rmse = np.array([model[clf]['neg_root_mean_squared_error'] for clf in models]) * -1
    precision = np.array([model[clf]['r2'] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_rmse = np.mean(rmse, axis=1)
    mean_precision = np.mean(precision, axis=1)

    # Creazione del grafico a barre per RMSE
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(models))
    plt.bar(index, mean_rmse, bar_width, label='neg_root_mean_squared_error')
    plt.xlabel('Modelli')
    plt.ylabel('RMSE medio')
    plt.title('RMSE medio per ogni modello')
    plt.xticks(index, models)
    plt.legend()
    plt.show()

    # Creazione del grafico a barre per R2
    plt.figure(figsize=(12, 6))
    plt.bar(index, mean_precision, bar_width, label='r2', color='orange')
    plt.xlabel('Modelli')
    plt.ylabel('R2 medio')
    plt.title('R2 medio per ogni modello')
    plt.xticks(index, models)
    plt.legend()
    plt.show()

