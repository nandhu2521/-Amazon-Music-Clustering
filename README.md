# -Amazon-Music-Clustering
Unsupervised ML project that clusters Amazon Music tracks by audio features (tempo, energy, danceability) using K-Means &amp; DBSCAN. Includes EDA, PCA visualization, and an interactive Streamlit app for real-time cluster prediction. Perfect for playlist generation &amp; music recommendations!
Amazon Music Clustering Project
Python scikit-learn Streamlit

An unsupervised machine learning project that automatically categorizes music tracks into meaningful clusters based on their audio characteristics using K-Means and DBSCAN clustering algorithms.

📋 Table of Contents
Problem Statement
Business Use Cases
Features
Tech Stack
Project Structure
Installation
Usage
Methodology
Results
Cluster Insights
Deployment
🎯 Problem Statement
With millions of songs available on platforms like Amazon Music, manually categorizing tracks into genres is impractical. This project automatically groups similar songs based on their audio characteristics using clustering techniques, organizing songs into meaningful clusters representing different musical genres or moods—without any prior labels.

💼 Business Use Cases
Personalized Playlist Curation: Automatically group songs that sound similar to enhance playlist generation
Improved Song Discovery: Suggest similar tracks to users based on their preferred audio profile
Artist Analysis: Help artists and producers identify competitive songs in the same audio cluster
Market Segmentation: Streaming platforms can analyze user listening patterns and optimize recommendations
✨ Features
Data Exploration & Preprocessing: Comprehensive EDA, handling missing values, and feature scaling
Multiple Clustering Algorithms: K-Means and DBSCAN implementation
Optimal Cluster Selection: Elbow method and Silhouette score analysis
Dimensionality Reduction: PCA visualization for 2D cluster representation
Model Persistence: Trained models saved for production use
Interactive Web App: Streamlit-based predictor for real-time cluster prediction
Comprehensive Analysis: Cluster profiling and genre inference
🛠️ Tech Stack
Languages & Libraries:

Python 3.8+
Pandas, NumPy - Data manipulation
Scikit-learn - Machine learning algorithms
Matplotlib - Data visualization
Joblib - Model serialization
Streamlit - Web application framework
Algorithms:

K-Means Clustering
DBSCAN (Density-Based Spatial Clustering)
Principal Component Analysis (PCA)
StandardScaler for feature normalization
📁 Project Structure
amazon-music-clustering/
│
├── amazon_music.ipynb          # Main Jupyter notebook with analysis
├── amazon_app.py               # Streamlit web application
├── single_genre_artists.csv    # Dataset
├── kmeans_model.pkl            # Trained K-Means model
├── scaler.pkl                  # Fitted StandardScaler
├── final_clustered_songs.csv   # Output with cluster labels
├── Amazon-Music-Clustering.pdf # Project documentation
└── README.md                   # This file
🚀 Installation
Prerequisites
Python 3.8 or higher
pip package manager
Steps
Clone the repository
git clone https://github.com/Deepak-Manian/amazon-music-clustering.git
cd amazon-music-clustering
Install required packages
pip install pandas numpy scikit-learn matplotlib joblib streamlit
Verify dataset Ensure single_genre_artists.csv is in the project directory.
💻 Usage
Running the Jupyter Notebook
jupyter notebook amazon_music.ipynb
Execute cells sequentially to:

Load and explore the dataset
Preprocess and scale features
Perform clustering analysis
Visualize results
Export clustered data
Running the Streamlit App
streamlit run amazon_app.py
The app will launch in your browser where you can:

Adjust audio feature sliders (danceability, energy, loudness, etc.)
Get real-time cluster predictions
Visualize where your input falls in the PCA space
🔬 Methodology
1. Data Exploration & Preprocessing
Loaded single_genre_artists.csv dataset
Checked for null values, duplicates, and data types
Dropped non-numeric columns: id_songs, name_song, name_artists
Selected 10 audio features for clustering
2. Feature Selection
Selected Features:

danceability - How suitable a track is for dancing
energy - Intensity and activity measure
loudness - Overall loudness in decibels
speechiness - Presence of spoken words
acousticness - Confidence measure of acoustic sound
instrumentalness - Predicts whether a track contains vocals
liveness - Presence of an audience
valence - Musical positiveness
tempo - Beats per minute (BPM)
duration_ms - Track length in milliseconds
3. Feature Scaling
Applied StandardScaler to normalize all features to the same scale (mean=0, std=1) for distance-based clustering.

4. Optimal Cluster Selection
Elbow Method: Plotted WCSS (Within-Cluster Sum of Squares) for k=1 to 10
Silhouette Score: Evaluated cluster quality for k=2 to 8
Optimal k=3 clusters selected
5. Clustering Algorithms
K-Means Clustering
Applied with n_clusters=3
Assigned cluster labels to all songs
Computed cluster centroids
DBSCAN Clustering
Used as an alternative approach
Parameters: eps=0.5, min_samples=5
Identified noise/outliers effectively
6. Dimensionality Reduction
Applied PCA with 2 components for visualization
Reduced 10D feature space to 2D while preserving variance
7. Model Persistence
Saved trained K-Means model: kmeans_model.pkl
Saved fitted scaler: scaler.pkl
Exported final dataset: final_clustered_songs.csv
📊 Results
Cluster Distribution (K-Means with k=3)
Cluster	Description	Key Characteristics
Cluster 0	Chill Acoustic	Low energy, high acousticness, calm vibe
Cluster 1	Party Tracks	High danceability, high energy, high tempo, high valence
Cluster 2	Rap/Live Recordings	High danceability, medium energy, high speechiness, high liveness
Evaluation Metrics
Silhouette Score Analysis: Performed for k=2 to 8 to determine optimal clustering
WCSS (Inertia): Decreased with increasing k, elbow observed at k=3
Cluster Visualization: 2D PCA projection showing clear separation
🎵 Cluster Insights
Cluster 0: Chill Acoustic
Mood: Relaxing, calm, introspective
Audio Profile:
Low energy and loudness
High acousticness
Low tempo
Use Case: Background music, study playlists, meditation
Cluster 1: Party Tracks
Mood: Energetic, upbeat, positive
Audio Profile:
High danceability and energy
High valence (positive)
High tempo
Use Case: Workout playlists, parties, dance clubs
Cluster 2: Rap/Live Recordings
Mood: Vocal-heavy, dynamic, live atmosphere
Audio Profile:
High speechiness
High liveness
Medium-high energy
Use Case: Hip-hop playlists, live concert recordings, spoken word
🌐 Deployment
Streamlit Web Application
The project includes a production-ready Streamlit app (amazon_app.py) that:

Loads pre-trained models (scaler and K-Means)
Accepts user input via interactive sliders for all 10 audio features
Predicts cluster in real-time
Displays cluster interpretation:
Cluster 0 → "Energetic Dance Tracks"
Cluster 1 → "Balanced Vocal Songs"
Cluster 2 → "Calm Acoustic"
Visualizes prediction on PCA scatter plot with the input point highlighted in red
Running in Production
streamlit run amazon_app.py --server.port 8501
📚 Skills Demonstrated
Data Exploration & Cleaning
Feature Engineering & Selection
Data Normalization (StandardScaler)
K-Means Clustering
DBSCAN Clustering
Elbow Method
Silhouette Score Analysis
Principal Component Analysis (PCA)
Cluster Visualization
Model Persistence (Joblib)
Web Application Development (Streamlit)
Data Storytelling
Python (Pandas, NumPy, scikit-learn, Matplotlib)
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
This project is licensed under the MIT License.

👤 Author
nandhini
Data Science & Machine Learning Engineer

🙏 Acknowledgments
Dataset: Amazon Music audio features dataset
Libraries: scikit-learn, Pandas, NumPy, Streamlit
Inspiration: Music recommendation systems and unsupervised learning applications
⭐ If you found this project helpful, please give it a star!
