Amazon Music Clustering 
Overview: 
        Domain : Music Analytics / Unsupervised Machine Learning
A Unsupervised  Machine learning project , In this project we are going to analyze patterns in features such as tempo, energy, danceability, and more and organize songs into meaningful clusters. 


Project Deliverables: 
* Exploring  & Understanding The Data
      * Correcting the data types 
      * Correcting the columns names 
      * Dropping the unwanted column 
* Preprocessing :
      * handling missing values, 
      * duplicates values, 
      * and feature scaling
* Clustering implementation using multiple clustering Algorithm
      * KMeans
      * DBSCAN 
      * Hierarchical clustering(agglomerative)
* Visualization 
      * Barchart
      * Boxplot
      * Distribution plot
* Key insights 
      * Chill acoustic 
      * High Energy Party
      * Party Tracks Rap and Live recording 






Business Use Cases:
* Personalized Playlist Curation:
 Automatically group songs that sound similar to enhance playlist generation.

* Improved Song Discovery:
 Suggest similar tracks to users based on their preferred audio profile.

* Artist Analysis:
 Help artists and producers identify competitive songs in the same audio cluster.

* Market Segmentation:
 Streaming platforms can use clusters to analyze user listening patterns and optimize recommendations or promotions.   


Technology Used : 
   * Python 3.13 ,
   * Pandas 2.3.3 ,
   * Matplotlib ,
   * Seaborn ,
   * Streamlit ,
   * Pickle
   * Scikit-learn 1.8.0
   * Unsupervised Machine Learning Algorithm,
   * KMeans,
   * DBSCAN,
   * Hierarchical Cluster Algorithm. 
Feature Selection: 
   * Danceability
   * Energy
   * Loudness
   * Speechiness
   * Acousticness
   * Instrumentalness
   * Liveness
   * Valence
   * Tempo
   * duration_ms




Run Application:  
   * Created Virtual Machine And activate them
   * Run the Jupiter notebook 
   * Run the python py file It will be redirected to web Browser.
Feature Scaling:
        Applied StandardScaler to normalize all features to the same scale
        (mean = 0, std = 1) for Distance-Based clustering.
Key Insights : 
        Cluster A : HighDanceability, High Energy, High Valence, High Tempo, 
                       Low Acoustic → High Energy Party 
        Cluster B: High Danceability, Medium to High Energy Liveness →
                      “Party Tracks Rap and Live Recordings” 
        Cluster C: Low Energy, High Acoustic, low Tempo → “Chill Acoustic” 
  





Optimal clutter Using Elbow Method : 
 Here I tried many values to k like, 1 to 9 and getting inertia values(WCSS) within the cluster sum of squares. 
   1. Elbow method is Graph of distortion score WCSSvs k  
   2. It is looking like elbow 
   3. Increase the number of clusters decreasing WCSS
   4. 3 is the optimal cluster for this dataset. 


  

Here  we can see after 3 there is no sharp decrease. K = 3 is the optimal cluster 
METHODOLOGY  ⚓: 
        KMeans, DBSCAN, Hierarchical (Aggolomerative) 
KMeans Clustering 👍: 
   1. Identified optional cluster using Elbow plot 
   2. Three(3) is the optimal cluster for the given data set 
   3. Finally We clustered the data, like chill Acoustic, Rap and Live, Energy and high          danceability songs. 
   4. Silhoustics score is 0.24 is very low performance in this dataset, indicates overlapping clusters


DBSCAN Clustering  💹: 
   * Identified outliers 
   * In this case no need to specified clusters 
   * Model automatically founded by itself 
   * Given eps = 2.3, min_samples = 5  
Hierarchical Clustering(Agglomerative) 📊 : 
   * Here we used linkage ‘ward’ method 
   * Drawn Dentrogram like tree kind of structure 
   * Dengrogram is bottom to top approach 


Model_Deployment🌐:
   * 10 input Features taken by the user 
   * Then those inputs are Scaled and given to the model 
   * Model is predicted which type of cluster Based on the user input  
                Cluster 0 → "Energetic Dance Tracks"
Cluster 1 → "Balanced Vocal Songs"
Cluster 2 → "Calm Acoustic" 


    Finally the input Features colored by red color with existing Scatter plot
