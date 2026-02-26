import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import streamlit as st 
import pickle as pkl 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

# Creating the streamlit page: 


st.set_page_config(
    layout = 'wide',
    page_icon = ":smile:",
    page_title = "Amazon Music Clustring"
) 
col11,col12 = st.columns([8,2])
with col11:
    st.markdown(
        "<h1 style='text-align: center; color: " \
        "black;'>Amazon Music Clustring </h1>", 
        
        unsafe_allow_html=True) 
    st.markdown("""
        <style>
        [data-testid=stAppViewContainer] {
        background-color: #88c9fa
             
        
        }
        </style>
        """, unsafe_allow_html=True)
    
with col12:
    st.image("C:/guvi/project_4/music-logo-design.jpg", width = 200)

st.markdown("""
        <style>
        [data-testid=stSidebar] {
        background-color: #5494DA
        } 
         div.row-widget.stRadio > div{
        flex-direction:column;
        }
        /* Target the text labels next to the radio buttons */
        div.stRadio label p {
        font-size: 20px !important;
        font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)



with st.sidebar.title("pages"):

    
    
    options = ["Model_Deployment", 'Data']
    page = st.radio("",options)

if page == 'Model_Deployment':
    
    st.markdown("""
    <style>
    /* Targets the label of st.text_input, st.number_input, st.text_area, etc. */
    .stTextInput p, .stNumberInput p, .stTextArea p, .stSelectbox p {
        font-size: 20px; /* Adjust the size as needed */
        font-weight: bold; /* Optional: make the label bold */
    }
    </style>
    """, unsafe_allow_html=True) 

    st.markdown("""
    <style>
    div[data-testid="stSlider"] label p {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4,col5 = st.columns([1,1,1,1,1])
    with col1:
        danceability = st.slider("Enter the danceability :", min_value = 0.0, max_value = 1.0, step = 0.01)
    with col2:
        energy = st.slider("Enter the energy :", min_value = 0.0, max_value = 1.0, step = 0.01)
    with col3:
        loudness = st.slider("Enter the loudness :", min_value = -50.0, max_value = 1.0, step = 0.01)
    with col4: 
        speechiness = st.slider("Enter the speechiness :", min_value = 0.0, max_value = 1.0, step = 0.01) 
    with col5:
        acousticness = st.slider("Enter the acousticness :", min_value = 0.0, max_value = 1.0, step = 0.01)
    col6,col7,col8,col9,col10 = st.columns([1,1,1,1,1]) 
    with col6:
        instrumentalness = st.slider("Enter the instrumentalness :", min_value = 0.0, max_value = 1.0, step = 0.01) 
    with col7: 
        liveness = st.slider("Enter the liveness :", min_value = 0.0, max_value = 1.0, step = 0.01) 
    with col8:
        valence = st.slider("Enter the valence :", min_value = 0.0, max_value = 1.0, step = 0.01) 
    with col9:
        tempo = st.slider("Enter the tempo :", min_value = 0.0, max_value = 100000.0, step = 0.01)
    with col10: 
        duration_Ms = st.slider("Enter the duration_Ms :", min_value = 0.0, max_value = 1000000.0, step = 0.01) 
    features = {
        'Danceability' :[danceability],
        'Energy' : [energy],
        'Loudness' :[loudness],
        'Speechiness': [speechiness],
        'Acousticness': [acousticness],
        'Instrumentalness': [instrumentalness],
        'Liveness' : [liveness],
        'Valence' : [valence],
        'Tempo' : [tempo],
        'Duration_Ms' : [duration_Ms]
    } 

    # Making the input Features like Dataframe format:
    
    st.markdown(
        '<h1 <p style="font-size:30px; color:black;"> Input Features Dataframe </h2></p>',
        unsafe_allow_html=True
        )
                                                                    
    input_df = pd.DataFrame(features)
    st.dataframe(input_df)
    
    if st.button('Click me'):
       
        # Loading Tll Saved Models:

        with open("C:/guvi/project_4/kmeans.pkl", "rb") as f:
            model = pkl.load(f) 
        with open("C:/guvi/project_4/pca.pkl", "rb") as f:
            pca_model = pkl.load(f)
        with open("C:/guvi/project_4/scaler_model.pkl", "rb") as file:
            scaler = pkl.load(file)

        st.write(model)
        
        
        scaled_df1 = scaler.transform(input_df)
        cluster = model.predict(scaled_df1)[0]  
        # PCA -principal component Analysis for user input:
        pca_df1=pd.DataFrame(pca_model.transform(scaled_df1))
        
        st.write("Predicted cluster :", cluster)
        if cluster == 1:
            st.info("High Energy party track")
        elif cluster == 2:
            st.info("Play Tracks Rap and Live Recordings")
        elif cluster == 0: 
            st.info("Chill Acoustic!") 

        
         
       
        #pca_model.explained_variance_ratio_

        df = pd.read_csv("C:/guvi/project_4/FinalizeData.csv", index_col = 0) 
        df1 = df.drop('Clusters', axis = 1) 
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df1) 
        pca = PCA(n_components=2)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_df))
        pca_df['Clusters'] = model.labels_ 
        #st.dataframe(pca_df)
        
        clusterA = pca_df[pca_df['Clusters'] == 0] 
        clusterB = pca_df[pca_df['Clusters'] == 1]
        clusterC = pca_df[pca_df['Clusters'] == 2]
        
        fig, ax = plt.subplots(figsize = (5, 3))
        ax.scatter(clusterA[0], clusterA[1], c = 'yellow', label = 'Chill acoustic', s = 10) 
        ax.scatter(clusterB[0], clusterB[1], c = 'green', label = 'High Energy Party', s = 10) 
        ax.scatter(clusterC[0], clusterC[1], c = clusterC['Clusters'], label = 'Rap and Live Recordings', s = 10)  
        ax.scatter(pca_df1[0], pca_df1[1], c = 'red', s = 10) 
        plt.legend(loc = 'upper left', fontsize = 5)
        plt.title("Amazon Music clusters") 
        
        st.pyplot(fig)
        
    


elif page == "Data" :
    df = pd.read_csv("C:/guvi/project_4/FinalizeData.csv", index_col = 0) 
    st.write("Finalized Data: ")
    st.dataframe(df)
