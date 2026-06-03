import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)
        reference_date = pd.Timestamp.now()
        rfm = df.groupby('CustomerId').agg(
            recency=('TransactionStartTime', lambda x: (reference_date - x.max()).days),
            frequency=('TransactionId', 'count'),
            monetary=('Amount', lambda x: x[x > 0].sum())
        ).reset_index()
        rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
        prod_mode = df.groupby('CustomerId')['ProductCategory'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
        prod_mode.columns = ['CustomerId', 'ProductCategory']
        channel_mode = df.groupby('CustomerId')['ChannelId'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
        channel_mode.columns = ['CustomerId', 'ChannelId']
        result = rfm.merge(prod_mode, on='CustomerId').merge(channel_mode, on='CustomerId')
        return result

def create_preprocessor():
    numeric_features = ['recency', 'frequency', 'monetary']
    categorical_features = ['ProductCategory', 'ChannelId']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def build_full_pipeline():
    pipeline = Pipeline(steps=[
        ('aggregator', CustomerAggregator()),
        ('preprocessor', create_preprocessor())
    ])
    return pipeline

def assign_high_risk_label(customer_df):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(customer_df[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    customer_df = customer_df.copy()
    customer_df['cluster'] = clusters
    cluster_summary = customer_df.groupby('cluster')[['frequency', 'monetary']].mean()
    high_risk_cluster = cluster_summary['frequency'].idxmin()
    customer_df['is_high_risk'] = (customer_df['cluster'] == high_risk_cluster).astype(int)
    return customer_df[['CustomerId', 'is_high_risk']]
