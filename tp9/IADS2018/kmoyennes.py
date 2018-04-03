# -*- coding: utf-8 -*-

# ------------------------------------------------
# package IADS2018
# UE 3I026 "IA et Data Science" -- 2017-2018
#
# Module kmoyennes.py:
# Fonctions pour le clustering
# ------------------------------------------------

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(df):
    matrice = df.as_matrix();#recup matrice
    dictionnary = dict()
    column_names = df.columns.values
    for i in range(len(matrice[0])):
        c_min = matrice[:,i].min()
        c_max = matrice[:,i].max() 
        dictionnary[column_names[i]] = (matrice[:,i] - c_min) / (c_max - c_min)
    dataframe = pd.DataFrame(dictionnary)
    return dataframe
# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(s1, s2):
    # on suppose que les deux series ont la meme dimension
    somme = 0
    dim = len(s1)
    for i in range(dim):
        somme += np.power(s1[i] - s2[i], 2)
    return np.sqrt(somme)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    columns = df.columns.values
    dim = len(columns)
    centro = dict()
    
    for i in columns:
        centro[i] = [np.mean(df[i])]
    return pd.DataFrame(centro)
# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    #recup la valeur du centroide du cluster
    ck = centroide(df)
    #pr chaque exemple(ligne) calculer sa dist de ck
    jk = 0
    # nombre de lignes
    nb_exemple = df.shape[0]
    for i in range(nb_exemple):
        d = dist_vect(df.iloc[i], ck.iloc[0])
        jk += np.power(d, 2)
    
    return jk
# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(k, df):
    n = df.shape[0]
    dim = df.shape[1]
    df_indices = list(df.index.values)
    k_indices = random.sample(df_indices, k)#tirage sans remise
    
    dico = dict()
    #recup des lines qui nous interessent
    lines = df.iloc[k_indices]
    lines.index = range(len(k_indices))#changement d'indices
    for c in df.columns.values:
#         print(lines[c].reindex)
        dico[c] = lines[c]
    return pd.DataFrame(dico)
# -------
# ************************* Recopier ici la fonction plus_proche()
import operator
def plus_proche(e, df):
    dico= dict()
    for i in range(df.shape[0]):
        s = df.iloc[i]
        dico[s.name] = dist_vect(s, e)
    #tri par valeur
    dico = sorted(dico.items(), key=operator.itemgetter(1))
#     print(dico)
    return dico[0][0]
# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(df, centroides):
    size = df.shape[0]
    matrice= dict()
    affectation = [-1 for i in range(size)]
    for i in range(size):
        affectation[i] = plus_proche(df.iloc[i], centroides)
    for i in range(centroides.shape[0]):
        matrice[i] = [x for x in range(len(affectation)) if affectation[x] == i]
    
    return matrice
# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(df, matrice):
    centroides = []
    for l in matrice.values():
        tmp_df = df.iloc[l]
        centroides.append(centroide(tmp_df))
    dico = dict()
#     print (centroides)
    for c in df.columns.values:
        dico[c]=[ct[c][0] for ct in centroides]
#     print (dico)
    return pd.DataFrame(dico)
# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(df, matrice):
    inertie = 0
    
    for c in matrice.values():
        tmp_c_df = df.iloc[c]
        inertie += inertie_cluster(tmp_c_df)
    return inertie
# -------
# ************************* Recopier ici la fonction kmoyennes()
#renvoie des centroides et une matrice d'affectation
def kmoyennes(k, df, epsilon, iter_max):
    # ensemble de centroides de depart
    centroides = initialisation(k, df)
    cpt = 0
    inert_t = 0
    inert_t_plus = 0
    diff_inert = 0
    while(cpt < iter_max ):
        # affecter chaque exemple au cluster le plus proche
        matrice = affecte_cluster(df, centroides)
#         print(matrice)
        #nouveaux centroides
        centroides = nouveaux_centroides(df, matrice)
        
        #en case de convergeance arreter
        inert_t_plus = inertie_globale(df, matrice)
        diff_inert = abs(inert_t_plus - inert_t)
        print ("Iteration ", cpt+1, "\tInertie: ", inert_t_plus, "\tDifference: ", diff_inert)
        if abs(diff_inert) < epsilon:
            break
        
        #incrementer les vbles
        inert_t = inert_t_plus
        cpt += 1
    
    return centroides, matrice
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
from matplotlib import colors
def affiche_resultat(df, centroides, matrice):
    labels = df.columns.values
    x = labels[0]
    y = labels[1]
    for l in matrice.values():
        tmp_df = df.iloc[l]
        c = np.random.rand(3)
        plt.scatter(tmp_df[x], tmp_df[y], color=c)
    plt.scatter(centroides[x],centroides[y],color='r',marker='x')
# -------
