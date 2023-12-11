import streamlit as st
import lime.lime_tabular
import shap
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
shap.initjs()

url = 'https://raw.githubusercontent.com/GPT05/AMII/main/BD_LANZAMIENTOS_2019_2021.csv'

data = pd.read_csv(url, encoding= 'latin')

# Configurar el tema de la página
st.set_page_config(
    page_title="Modelo predicción Skip Rate en Spotify",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Aprendizaje de Máquina II")
    st.subheader("Integrantes")
    st.write("Pedro Antonio Domínguez Bernal \n")
    st.write("Jorge Carlos Duarte Gutiérrez \n")
    st.write("Gabriela Pérez Toriz  \n")

with st.container():
    st.title("Modelo de Predicción: ¿Escucharán o Saltarán la Canción?")


    #Interfaz para seleccionar modo
    mode_selection = st.radio("Selecciona el modo:", ["Automático", "Manual"])
    if mode_selection == "Automático":
      st.subheader("Selecciona un artista")
      selected_artist = st.selectbox("Artista:", data['PRIMARY_ARTIST'].unique())
      songs_by_artist = data[data['PRIMARY_ARTIST'] == selected_artist]

      # Interfaz para seleccionar canción
      st.subheader("Selecciona una canción del artista")
      selected_song = st.selectbox("Canción:", songs_by_artist['SONG'].unique())

      # Obtener valores de la canción seleccionada
      selected_song_data = songs_by_artist[songs_by_artist['SONG'] == selected_song].iloc[0]
      st.subheader("Estos son los features de la canción elegida")
      st.write('[¿Qué significan cada una de las features?](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)')

      # Se colocan  los valores de la canción elegida en los controles deslizantes
      feature1 = st.slider('DURACION (en segundos)', min_value=0, max_value=600, value=selected_song_data['DURACION'])
      feature2 = st.slider('ACOUSTICNESS', min_value=0.0, max_value=1.0, value=selected_song_data['ACOUSTICNESS'], step=0.01)
      feature3 = st.slider('DANCEABILITY', min_value=0.0, max_value=1.0, value=selected_song_data['DANCEABILITY'], step = 0.01)
      feature4 = st.slider('ENERGY', min_value=0.0, max_value=1.0, value=selected_song_data['ENERGY'], step = 0.01)
      feature5 = st.slider('LIVENESS', min_value=0.0, max_value=1.0, value=selected_song_data['LIVENESS'], step = 0.01)
      feature6 = st.slider('LOUDNESS (decibeles)', min_value=-60, max_value=0, value=int(selected_song_data['LOUDNESS']))
      feature7 = st.slider('SPEECHINESS', min_value=0.0, max_value=1.0, value=selected_song_data['SPEECHINESS'], step = 0.01)
      feature8 = st.slider('TEMPO (BPM)', min_value=0, max_value=500, value=int(selected_song_data['TEMPO']))
      feature9 = st.slider('VALENCE', min_value=0.0, max_value=1.0, value=selected_song_data['VALENCE'], step = 0.01)
    else:
      # Modo manual
      st.subheader("Modo Manual - Ajusta los controles deslizantes")
      feature1 = st.slider('DURACION (en segundos)', min_value=0, max_value=600, value=0)
      feature2 = st.slider('ACOUSTICNESS', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)
      feature3 = st.slider('DANCEABILITY', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)
      feature4 = st.slider('ENERGY', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)
      feature5 = st.slider('LIVENESS', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)
      feature6 = st.slider('LOUDNESS (decibeles)', min_value=-60, max_value=0, value=-60)
      feature7 = st.slider('SPEECHINESS', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)
      feature8 = st.slider('TEMPO (BPM)', min_value=0, max_value=500, value=0)
      feature9 = st.slider('VALENCE', min_value=0.0, max_value=1.0, value=0.0, step = 0.01)

urlm = 'https://raw.githubusercontent.com/GPT05/AMII/main/clf_model.pkl'
response = requests.get(urlm)
loaded_model = pickle.loads(response)

#def cargar_modelo():
   #with open('clf_model.pkl', 'rb') as file:
       #model = pickle.load(file)
   #return model

#loaded_model = cargar_modelo()

if st.button('Calcular'):
    input_data = pd.DataFrame({'DURACION': [feature1], 'ACOUSTICNESS': [feature2], 'DANCEABILITY': [feature3], 'ENERGY': [feature4], 'LIVENESS': [feature5],'LOUDNESS': [feature6],'SPEECHINESS': [feature7],'TEMPO': [feature8],'VALENCE': [feature9]})
    prediction = loaded_model.predict(input_data)
    prediction_proba = loaded_model.predict_proba(input_data)
    prediction_value = prediction_proba[0][1] *100 #Multiplicamos por 100 para obtener porcentaje
    st.write(f'La probabilidad de que la canción sea reproducida es: {prediction_value:.2f}%')

    #st.write("---")
    #st.subheader('Explicación de Lime')
    #explainer_lime = lime.lime_tabular.LimeTabularExplainer(input_data.values,feature_names=input_data.columns.tolist(), class_names=['No Reproducida', 'Reproducida'],discretize_continuous=True)
    #sample_index = 0
    #instance_to_explain = input_data.iloc[sample_index]
    #exp = explainer_lime.explain_instance(instance_to_explain.values, loaded_model.predict_proba, num_features=5, top_labels=1)
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #lime_plot = exp.as_pyplot_figure()
    #st.write('Local explanation LIME:')
    #st.pyplot(lime_plot)

    st.write("---")
    st.subheader('Explicación de Shap')
    explainer_shap = shap.TreeExplainer(loaded_model)
    shap_values = explainer_shap.shap_values(input_data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    force_plot = shap.force_plot(explainer_shap.expected_value, shap_values[0], input_data.iloc[0], matplotlib = True)
    st.write('Force Plot Shap:')
    st.pyplot(force_plot)
