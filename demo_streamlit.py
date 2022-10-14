# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:34:47 2022

@author: elodi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error
from dateutil.relativedelta import relativedelta
from prophet import Prophet

st.set_page_config(page_title='Pyc énergétique', layout='wide')

st.title("Pyc énergétique")

pages=["Le projet", "Prévision par mois avec SARIMA", "Prévision par jour avec Prophet", "Prévision par heure avec Prophet"]

page=st.sidebar.radio("Aller vers", pages)

df=pd.read_csv("http://linky94.free.fr/Data/conso_only.csv", sep=",")
#df=df[["Code INSEE région", "Région", "Date", "Heure", "Date - Heure", "Consommation (MW)"]]
#df=df.dropna(axis=0, subset=["Consommation (MW)"])
#df=df.rename(columns = {'Consommation (MW)':'Consommation'})
df.Date=pd.to_datetime(df.Date)
df=df.set_index("Date")
    
if page == pages[0]:
    st.markdown("## **Objectif**") 
    st.write("Constater le phasage entre la consommation et la production énergétique au niveau national, et au niveau régional (risque de black-out notamment) :")
    """
    * Analyse au niveau régional pour en déduire une prévision de consommation 
   * Analyse par filière de production : énergie nucléaire / énergies renouvelables 
    * Focus sur les énergies renouvelables (lieux d’implantation)
    """
    st.markdown("## **Source des données**")
    st.markdown("La source de données est celle de l’ODRE (Open Data Réseaux Energies) et a été téléchargée [depuis cette page](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure).")

    st.markdown("## **Temporalité de la data**")
    st.line_chart(df.Consommation["2016":"2022"].resample("D").mean())
        
elif page==pages[1]:

    st.markdown("### **Prévisions pour 2022-2023 pour une région au choix**")
    region = st.selectbox(label = "Choix de la région", options=["Île-de-France","Centre-Val de Loire","Bourgogne-Franche-Comté", "Normandie", "Hauts-de-France","Grand Est","Pays de la Loire","Bretagne", "Nouvelle-Aquitaine","Occitanie","Auvergne-Rhône-Alpes","Provence-Alpes-Côte d'Azur"])
    hypothesis = st.selectbox(label = "Hypothèse n°", options=[0,1])
    st.write("Les données d'entraînement prennent en compte la totalité des données (par mois), jusqu'en 2019 (hypothèse 0), ou 2021 (hypothèse 1). Les données 2022 (jusque fin mai) sont les données de test.")
    def sarima_region(region, hypothesis):
#        dic={11:"Ile de France", 24:"Centre Val de Loire", 27:"Bourgogne Franche Comté", 28:"Normandie", 32:"Hauts de France", 44:"Grand Est", 52:"Pays de la Loire", 53:"Bretagne", 75:"Nouvelle Aquitaine", 76:"Occitanie", 84:"Auvergne Rhône Alpes", 93:"Provence Alpes Côte d'Azur"}
        if hypothesis==0:
            train_data=df[df["Région"]==region].resample("M").mean().Consommation[:"2019"]
            test_data=df[df["Région"]==region].resample("M").mean().Consommation["2022"]
        elif hypothesis==1:
            train_data=df[df["Région"]==region].resample("M").mean().Consommation[:"2021"]
            test_data=df[df["Région"]==region].resample("M").mean().Consommation["2022"]

        fit=auto_arima(train_data, start_p=1, start_q=1, max_p=3, max_q=3, m=12, d=1, max_d=3, trace=False, warnings=False, error_action="ignore")  
        model = SARIMAX(train_data, order=fit.get_params().get("order"), seasonal_order=fit.get_params().get("seasonal_order")).fit()
        predictions = model.predict(start=test_data.index[0], end=test_data.index[-1]+relativedelta(months=+12))

        fig, ax=plt.subplots(figsize=(15,5))
        ax.plot(predictions, label="Prédictions")
        ax.plot(df[df["Région"]==region].resample("M").mean().Consommation["2016":], label="Données réelles") 
        ax.set_ylabel("Puissance en MW")
        ax.legend()
        ax.set_title("Prédictions 2022/2023 pour la région " + str(region), weight="bold")
  
        st.pyplot(fig)

        mae=round(mean_absolute_error(test_data,predictions[:5]),2)
        erreur=round(mae/test_data.mean()*100,2)
        S=0
        for pred, value in zip(predictions[:5], test_data):
          S+=(pred - value)
        mr=round(S/5,2)
  
        return mae, erreur, mr

    a,b,c=sarima_region(region, hypothesis)

    st.write("La MAE est de", a, "soit environ", b, "% de marge d'erreur. \nLa moyenne des résidus est de", c,".")
    
elif page==pages[2]:
    
    st.markdown("### **Prévisions par jour pour une région au choix**")
    region = st.selectbox(label = "Choix de la région", options=["Île-de-France","Centre-Val de Loire","Bourgogne-Franche-Comté", "Normandie", "Hauts-de-France","Grand Est","Pays de la Loire","Bretagne", "Nouvelle-Aquitaine","Occitanie","Auvergne-Rhône-Alpes","Provence-Alpes-Côte d'Azur"])

    st.write("Les données d'entraînement correspondent aux données 2020-2021 (voir les dots sur le graphique), les prédictions sont ensuite réalisées pour l'ensemble de l'année 2022.")
    st.write("(Pour une raison inconnue, toutes les améliorations apportées au graphique (nom des axes, légende, etc...) n'ont pas fonctionné lors du déploiement en ligne alors qu'elles fonctionnaient en local. A  noter donc que les points en noir représentent les données d'observations, la ligne en bleu, les prédictions selon Prophet, avec en bleu clair l'intervalle de confiance.)")
    def prophet_day(region):
        train_data=df[df["Région"]==region]["Consommation"]["2020":"2021"].resample("D").mean().reset_index().rename(columns={"Date" : "ds", "Consommation" : "y"})
        prophet_model = Prophet(yearly_seasonality=True).fit(train_data)
        futur = prophet_model.make_future_dataframe(periods=30*12, freq='D')
        predictions=prophet_model.predict(futur)
            
  #Représentation graphique des data et prédictions
        fig = prophet_model.plot(predictions) 
        #ax = fig.add_subplot(111)
        plt.xlabel("")
        plt.ylabel("Puissance en MW")
        fig.set_size_inches(30, 10)
        
        plt.title("Prédictions pour 2022 en région " + str(region), weight="bold", fontsize=16)
       # ax.plot(df[df["Région"]==region].resample("D").mean().Consommation["2022"], color="red", label="Données réelles", ax=ax) 
        plt.legend(loc="upper right")

        st.pyplot(fig)
        
  #Calcul du MAE

        S=0
        S2=0
        for pred, value in zip(predictions[(predictions["ds"]>="2022-01-01")&(predictions["ds"]<="2022-05-31")]["yhat"], df[df["Région"]==region]["Consommation"]["2022"].resample("D").mean()) :
            S+= np.abs(pred - value)
            S2 += (pred - value)
        mae = round(S/151,2)
        mr = round(S2/151,2)
        erreur = round(mae/(df[df["Région"]==region]["Consommation"]["2022"].resample("D").mean().mean())*100,2)
            
        return mae, mr, erreur
 
    a,b,c = prophet_day(region)
    st.write("La MAE (sur les données 2022) est de", a, "soit environ", c,"% de marge d'erreur. \nLa moyenne des résidus est de", b,".")
         
elif page==pages[3]:
    
    st.markdown("### **Prévisions pour deux jours dans une région au choix en 2022**")
    st.write("On propose de prédire la consommation heure par heure pendant deux jours.")
    st.write("Les données d'entraînement correspondent au 10 jours précédents (du 12 au 22, même si Prophet opère un lissage des données, correspondant à la courbe 'Prédictions'), les prédictions sont faites sur les deux derniers jours.")

    region = st.selectbox(label = "Choix de la région", options=["Île-de-France","Centre-Val de Loire","Bourgogne-Franche-Comté", "Normandie", "Hauts-de-France","Grand Est","Pays de la Loire","Bretagne", "Nouvelle-Aquitaine","Occitanie","Auvergne-Rhône-Alpes","Provence-Alpes-Côte d'Azur"])
    month = st.selectbox(label = "Choix du mois", options=[1,2,3,4,5])
    def prophet_hour(region, month):

        train_data=df[df["Région"]==region][["Consommation", "Date - Heure"]]["2022-"+str(month)+"-12" : "2022-"+str(month)+"-22"].rename(columns={"Date - Heure" : "ds", "Consommation" : "y"}).reset_index().drop(columns="Date")
        train_data.ds=train_data.ds.apply(lambda x:x[:-6])

        prophet_model = Prophet(weekly_seasonality=True).fit(train_data) 
        futur = prophet_model.make_future_dataframe(periods=24*2, freq='H')
        predictions=prophet_model.predict(futur)

        real = df[df["Région"]==region][["Consommation", "Date - Heure"]]["2022-"+str(month)+"-12" : "2022-"+str(month)+"-24"]
        real["Date - Heure"] = pd.to_datetime(real["Date - Heure"])
   
        fig,ax = plt.subplots(figsize=(30,10))
        predictions.set_index("ds").yhat.plot(label="Prédictions")
        real.set_index(["Date - Heure"]).Consommation.plot(label="Données réelles", ax=ax)
        ax.set_title("Prédictions pour les 23 et 24/"+str(month)+" 2022 en région " + str(region), weight="bold", size=15)
        ax.set_ylabel("Puissance en MW")
        ax.set_xlabel("")
        ax.legend(loc="upper right")
        
        st.pyplot(fig)
        
        
    prophet_hour(region, month)

  
