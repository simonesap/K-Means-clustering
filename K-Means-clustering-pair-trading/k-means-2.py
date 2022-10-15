#Import ML libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from sklearn.preprocessing import StandardScaler
#Import Python libraries
import scipy.stats as scs
import numpy as np
import numpy.random as npr
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
from datetime import datetime
#Import and Analyze Data

#Download data from Yahoo Finance
import yfinance as yf
start = datetime(2020, 1, 1)
end = datetime(2022, 8, 26)
data = yf.download('EURUSD=X DX=F GC=F BTC-USD CL=F NG=F FTSEMIB.MI DAX ^VIX SPY', start=start, end=end)

#Cerco null value per evitare problemi
data.isnull().values.any()
False
#Calcolo il ritorno giornaliero e plotto
rets = data['Adj Close']/data['Adj Close'].shift(1)
rets = rets.dropna()
rets.hist(bins=50, figsize = (20, 10));

#Matrice annuallizzata della covarianza
covmatrix = rets.cov() * 252 #La covarianza di un asset con se stesso è la varianza (i.e. il quadrato della sua deviazione standard o volatilità)
#Visualizzo la matrice di covarianza
sns.clustermap(covmatrix, annot=True);

#Calcolo e visualizzo heatmap correlazione
corrmatrix = rets.corr()
plt.figure(figsize=(15, 15))
plt.title ('Correlation Matrix of Returns')
sns.heatmap(corrmatrix, vmax=1, square=True, annot=True, cmap='cubehelix');

#Build Clusters

#Calcoliamo il ritorno annuale e la volatilità delle stocks . Sono le due features del modello
model = pd.DataFrame()
#Ritorno annuale
model['Returns'] = data['Adj Close'].pct_change().mean()*252
#Volatilità
model['Volatility'] = data['Adj Close'].pct_change().std()*np.sqrt(252)
model.head()

#Standardizzazione features media 0 varianza unitaria
standardized_model = StandardScaler().fit_transform(model)
standardized_model = pd.DataFrame(standardized_model, columns=model.columns, index=model.index)
standardized_model.head()

#Build K-Means Clustering Model
k_means = KMeans(n_clusters=5, random_state=1) #Instanziamo K-Means con un tentativo di cluster
k_means.fit(standardized_model) # Usiamo il fit per trovare i clusters
pair_trades = model #Creiamo un dataframe nuovo per analizzare i pairs
pair_trades['Cluster'] = k_means.labels_ #otteniamo le labels (non hanno significato semantico)
pair_trades.sort_values(by=['Cluster']) #Ordiniamo i dati per label

#Visualize Clusters with Dendograms

dist = linkage(standardized_model, method='ward') #This one line creates hierarchical clusters. Ward minimizes the variance of merged clusters
plt.figure(figsize=(10,8))
dendrogram(dist, labels=standardized_model.index)
plt.title('Dendogram for Pairs Trading')
plt.show()

#Create Hierarchical Clusters
agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, linkage='ward') #Use distance metric to create clusters
agg_cluster.fit(standardized_model)
pair_trades_agg = model
pair_trades_agg['Cluster'] = agg_cluster.labels_
pair_trades_agg.sort_values(by=['Cluster'])

dist = linkage(standardized_model, method='average') #This one line creates hierarchical clusters. average minimizes the mean of merged clusters
plt.figure(figsize=(10,8))
dendrogram(dist, labels=standardized_model.index)
plt.title('Dendogram for Pairs Trading')
plt.show()

#Create Hierarchical Clusters
agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage='average') #Use distance metric to create clusters
agg_cluster.fit(standardized_model)
pair_trades_agg = model
pair_trades_agg['Cluster'] = agg_cluster.labels_
pair_trades_agg.sort_values(by=['Cluster'])