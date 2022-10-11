# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:38:30 2022

@author: elodi
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Pyc énergétique', layout='wide')

st.title("Pyc énergétique")

df=pd.read_csv("consoelec.csv", sep=";")
df=df.drop(columns=["TCO Thermique (%)", "TCH Thermique (%)", "TCO Nucléaire (%)","TCH Nucléaire (%)","TCO Eolien (%)","TCH Eolien (%)","TCO Solaire (%)","TCH Solaire (%)","TCO Hydraulique (%)","TCH Hydraulique (%)","TCO Bioénergies (%)","TCH Bioénergies (%)",] )
df=df.dropna(axis=0, subset=["Consommation (MW)"])
df=df.rename(columns = {'Consommation (MW)':'Consommation','Thermique (MW)': 'Thermique', 'Nucléaire (MW)': 'Nucléaire','Eolien (MW)':'Eolien', 'Solaire (MW)':'Solaire', 'Hydraulique (MW)':'Hydraulique', 'Pompage (MW)':'Pompage', 'Bioénergies (MW)':'Bioénergies', 'Ech. physiques (MW)':'Ech. physiques' })
df_sum=df.groupby(["Date","Heure"], as_index=False).agg({'Eolien' : "sum",'Hydraulique' : "sum", 'Nucléaire': "sum", 'Pompage' : "sum", 'Solaire':"sum", 'Thermique' : "sum",'Bioénergies':"sum"})
df_sum.Date=pd.to_datetime(df_sum.Date)
df_sum.Heure=pd.to_datetime(df_sum.Heure)
df_sum=df_sum.set_index("Date")

st.write("L'année 2018 a été arbitrairement choisie.")
st.write("L'analyse par heure concerne la journée du 25 mai 2018")

def graph():
    
    energie = st.multiselect(label="Type d'énergie", options=["Eolien", "Hydraulique", "Nucléaire", "Pompage", "Solaire", "Thermique", "Bioénergies"])    
    temp = st.select_slider("Choix de la temporalité",options=['Année', 'Mois', 'Jour'])
    if temp=="Année":
        chart_data=df_sum[energie]["2018"].resample("M").mean()
        st.line_chart(chart_data)
    elif temp=="Mois":
        chart_data=df_sum[energie]["2018-05"].resample("D").mean()
        st.line_chart(chart_data)
    elif temp=="Jour":
        chart_data=df_sum.loc["2018-05-25"]
        st.line_chart(chart_data, x="Heure", y=energie)
    
    
graph()