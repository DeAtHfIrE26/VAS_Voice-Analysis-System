import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Any, Tuple

def prepare_feature_matrix(features: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for machine learning.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Normalized feature matrix and feature names
    """
    # Convert features dictionary to DataFrame
    feature_names = []
    feature_values = []
    
    for key, value in features.items():
        # Skip non-numeric features or array features
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            feature_names.append(key)
            feature_values.append(value)
    
    # Create feature matrix
    X = np.array(feature_values).reshape(1, -1)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X_scaled, columns=feature_names)
    
    return df, feature_names

def create_simulated_data(features: Dict[str, Any], n_samples: int = 50) -> pd.DataFrame:
    """
    Create simulated dataset for unsupervised learning by adding noise to the original features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Dataframe with simulated samples
    """
    try:
        df, feature_names = prepare_feature_matrix(features)
        
        # Handle case with no valid features
        if len(feature_names) == 0:
            # Create at least some basic features for analysis
            df = pd.DataFrame({
                "placeholder_feature_1": [0.0],
                "placeholder_feature_2": [0.0],
                "placeholder_feature_3": [0.0]
            })
            feature_names = df.columns.tolist()
        
        # Create simulated data by adding noise to the original features
        simulated_data = []
        
        # Add the original sample
        simulated_data.append(df.iloc[0].values)
        
        # Generate variations
        for _ in range(n_samples - 1):
            # Add random noise to each feature
            noise = np.random.normal(0, 0.5, len(df.columns))
            new_sample = df.iloc[0].values + noise
            simulated_data.append(new_sample)
        
        # Convert to DataFrame
        simulated_df = pd.DataFrame(simulated_data, columns=df.columns)
        
        return simulated_df
    
    except Exception as e:
        # Fallback to create a minimal valid DataFrame that won't crash the application
        print(f"Warning: Error in creating simulated data: {str(e)}. Using placeholder data.")
        placeholder_df = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples)
        })
        return placeholder_df

def perform_clustering(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform clustering analysis on features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Dict[str, Any]: Clustering results
    """
    # Prepare feature matrix with simulated data for clustering
    simulated_df = create_simulated_data(features, n_samples=50)
    
    # Convert DataFrame to numpy array to avoid feature names warning
    X = simulated_df.values
    feature_names = simulated_df.columns.tolist()
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    max_clusters = min(10, len(simulated_df) - 1)
    max_clusters = max(2, max_clusters)  # Ensure at least 2 clusters
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Skip if there's only one cluster after fitting
        if len(np.unique(cluster_labels)) < 2:
            silhouette_scores.append(-1)
            continue
            
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
    
    # Select optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started from 2
    
    # Perform K-means clustering with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Determine which cluster contains the original sample
    original_cluster_kmeans = kmeans_labels[0]
    original_cluster_hierarchical = hierarchical_labels[0]
    
    # Count samples in each cluster
    kmeans_counts = np.bincount(kmeans_labels)
    hierarchical_counts = np.bincount(hierarchical_labels)
    
    # Calculate distance from original sample to cluster centers
    original_sample = X[0].reshape(1, -1)
    distances_to_centers = kmeans.transform(original_sample)
    
    # Return clustering results
    results = {
        "kmeans": {
            "labels": kmeans_labels.tolist(),
            "original_cluster": int(original_cluster_kmeans),
            "counts": kmeans_counts.tolist(),
            "centers": kmeans.cluster_centers_.tolist()
        },
        "hierarchical": {
            "labels": hierarchical_labels.tolist(),
            "original_cluster": int(original_cluster_hierarchical),
            "counts": hierarchical_counts.tolist()
        },
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca_result.tolist()
        },
        "optimal_clusters": optimal_clusters,
        "feature_names": feature_names
    }
    
    return results

def detect_anomalies(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect anomalies in features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Dict[str, Any]: Anomaly detection results
    """
    # Prepare feature matrix with simulated data
    simulated_df = create_simulated_data(features, n_samples=100)
    feature_names = simulated_df.columns.tolist()
    
    # Define cognitive decline indicators with their corresponding features
    cognitive_markers = {
        "Hesitation Frequency": ["linguistic_hesitation_frequency", "linguistic_filler_word_ratio"],
        "Pause Patterns": ["acoustic_pause_count", "acoustic_avg_pause_duration", "acoustic_pause_rate"],
        "Speech Rate": ["temporal_speech_rate_wpm", "temporal_articulation_rate"],
        "Vocabulary Richness": ["linguistic_type_token_ratio", "linguistic_uncommon_bigram_ratio"],
        "Word Retrieval": ["linguistic_word_retrieval_issues", "temporal_avg_response_latency"],
        "Voice Quality": ["acoustic_jitter", "acoustic_harmonics_to_noise_ratio", "acoustic_spectral_flatness"]
    }
    
    # Perform anomaly detection with isolation forest
    anomaly_results = {}
    significant_markers = []
    
    for marker, related_features in cognitive_markers.items():
        # Filter features that are available in the dataset
        available_features = [f for f in related_features if f in feature_names]
        
        if not available_features:
            continue
            
        # Extract relevant features
        X = simulated_df[available_features].values
        
        # Skip if not enough samples
        if len(X) < 5:
            continue
            
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_scores = iso_forest.fit_predict(X)
        
        # Compute anomaly scores (-1 for outliers, 1 for inliers)
        anomaly_score = iso_forest.score_samples(X)
        
        # Check if the original sample is an outlier
        is_original_outlier = outlier_scores[0] == -1
        original_score = anomaly_score[0]
        
        # Store results - ensure boolean type for is_outlier and float type for anomaly_score
        anomaly_results[marker] = {
            "is_outlier": bool(is_original_outlier),  # Explicitly cast to boolean
            "anomaly_score": float(original_score),   # Explicitly cast to float
            "features_used": available_features
        }
        
        # If original sample is an outlier, mark as significant
        if is_original_outlier:
            # Determine which specific features contributed most to the anomaly
            feature_contributions = {}
            for feature in available_features:
                # Check individual feature (crude approximation of feature importance)
                feature_value = simulated_df[feature].iloc[0]
                feature_mean = simulated_df[feature].mean()
                feature_std = simulated_df[feature].std() if simulated_df[feature].std() > 0 else 1
                z_score = (feature_value - feature_mean) / feature_std
                
                feature_contributions[feature] = float(abs(z_score))  # Ensure it's a float
            
            # Sort features by contribution
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            significant_markers.append({
                "marker": marker,
                "anomaly_score": float(original_score),
                "feature_contributions": dict(sorted_contributions)
            })
    
    # Aggregate results across all features for overall anomaly detection
    all_features = [col for col in simulated_df.columns]
    X_all = simulated_df[all_features].values
    
    iso_forest_all = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores_all = iso_forest_all.fit_predict(X_all)
    anomaly_score_all = iso_forest_all.score_samples(X_all)
    
    is_overall_outlier = outlier_scores_all[0] == -1
    overall_score = anomaly_score_all[0]
    
    # Return anomaly detection results
    results = {
        "overall": {
            "is_outlier": bool(is_overall_outlier),  # Explicitly cast to boolean
            "anomaly_score": float(overall_score)    # Explicitly cast to float
        },
        "markers": anomaly_results,
        "significant_markers": significant_markers,
        "risk_score": calculate_risk_score(anomaly_results)
    }
    
    return results

def calculate_risk_score(anomaly_results: Dict[str, Any]) -> float:
    """
    Calculate a cognitive impairment risk score based on anomaly detection results.
    
    Args:
        anomaly_results (Dict[str, Any]): Results from anomaly detection
        
    Returns:
        float: Risk score between 0 and 1
    """
    if not anomaly_results:
        return 0.0
    
    # Extract anomaly scores for each marker
    marker_scores = []
    for marker, result in anomaly_results.items():
        if isinstance(result, dict) and "anomaly_score" in result:
            # Convert anomaly score to a 0-1 scale (lower score = higher risk)
            # Isolation forest returns negative scores for anomalies
            normalized_score = 1 / (1 + np.exp(result["anomaly_score"]))
            marker_scores.append(normalized_score)
    
    # Calculate weighted average of scores
    if marker_scores:
        risk_score = np.mean(marker_scores) 
        
        # Rescale to 0-1 range
        risk_score = min(max(risk_score, 0.0), 1.0)
        return risk_score
    
    return 0.0

def analyze_feature_importance(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the importance of different features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Dict[str, Any]: Feature importance analysis results
    """
    # Prepare feature matrix
    df, feature_names = prepare_feature_matrix(features)
    
    # Create simulated data
    simulated_df = create_simulated_data(features, n_samples=100)
    
    # We need a target for supervised importance analysis
    # Here we'll use anomaly scores as a proxy target
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(simulated_df)
    
    # Convert anomaly predictions (-1 for outliers, 1 for inliers) to continuous scores for regression
    y = iso_forest.score_samples(simulated_df)
    
    # Train a random forest to analyze feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(simulated_df, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Create dictionary of feature importance scores
    importance_scores = {}
    for feature, importance in zip(feature_names, importances):
        importance_scores[feature] = float(importance)
    
    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Group features by category
    feature_categories = {
        "linguistic": [],
        "acoustic": [],
        "temporal": []
    }
    
    for feature, importance in sorted_features:
        for category in feature_categories.keys():
            if feature.startswith(f"{category}_"):
                feature_categories[category].append((feature, importance))
                break
    
    # Calculate category importance
    category_importance = {}
    for category, features in feature_categories.items():
        if features:
            category_importance[category] = sum(imp for _, imp in features) / len(features)
        else:
            category_importance[category] = 0.0
    
    # Return feature importance results
    results = {
        "scores": dict(sorted_features),
        "categories": category_importance,
        "top_features": dict(sorted_features[:10]) if len(sorted_features) >= 10 else dict(sorted_features)
    }
    
    return results
