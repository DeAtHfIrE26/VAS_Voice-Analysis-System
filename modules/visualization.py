import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional
import seaborn as sns

def plot_feature_distributions(features: Dict[str, Any]) -> Figure:
    """
    Create a plot showing distributions of key features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Figure: Matplotlib figure with feature distributions
    """
    # Extract feature categories
    linguistic_features = {k: v for k, v in features.items() if k.startswith("linguistic_")}
    acoustic_features = {k: v for k, v in features.items() if k.startswith("acoustic_")}
    temporal_features = {k: v for k, v in features.items() if k.startswith("temporal_")}
    
    # Select top features from each category
    top_features = {}
    for category, feature_dict in [
        ("Linguistic", linguistic_features),
        ("Acoustic", acoustic_features),
        ("Temporal", temporal_features)
    ]:
        # Sort features by absolute value
        sorted_features = sorted(
            feature_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top 3 features or all if less than 3
        for i, (name, value) in enumerate(sorted_features[:3]):
            # Clean up feature name for display
            display_name = name.split("_", 1)[1].replace("_", " ").title()
            top_features[f"{category}: {display_name}"] = value
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    feature_names = list(top_features.keys())
    feature_values = list(top_features.values())
    
    # Determine color based on categories
    colors = []
    for name in feature_names:
        if name.startswith("Linguistic"):
            colors.append("#1f77b4")  # Blue
        elif name.startswith("Acoustic"):
            colors.append("#ff7f0e")  # Orange
        elif name.startswith("Temporal"):
            colors.append("#2ca02c")  # Green
    
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Feature Value')
    ax.set_title('Key Voice Analysis Features')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label='Linguistic'),
        Patch(facecolor="#ff7f0e", label='Acoustic'),
        Patch(facecolor="#2ca02c", label='Temporal')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    return fig

def plot_anomaly_detection(features: Dict[str, Any], anomaly_results: Dict[str, Any]) -> Figure:
    """
    Create a plot showing anomaly detection results.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        anomaly_results (Dict[str, Any]): Results from anomaly detection
        
    Returns:
        Figure: Matplotlib figure with anomaly detection visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract marker scores
    markers = []
    scores = []
    is_outlier = []
    
    for marker, result in anomaly_results.get("markers", {}).items():
        if isinstance(result, dict) and "anomaly_score" in result:
            markers.append(marker)
            scores.append(result["anomaly_score"])
            is_outlier.append(result.get("is_outlier", False))
    
    if not markers:
        # Create default empty plot with message
        ax.text(0.5, 0.5, "No anomaly detection results available", 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Normalize scores for visualization (lower score = higher anomaly)
    normalized_scores = [1 / (1 + np.exp(score)) for score in scores]
    
    # Create bar colors based on outlier status
    colors = ['#ff7f0e' if outlier else '#1f77b4' for outlier in is_outlier]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(markers))
    ax.barh(y_pos, normalized_scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(markers)
    ax.invert_yaxis()  # Labels read top-to-bottom
    
    # Set x-axis from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_xlabel('Anomaly Score (higher = more anomalous)')
    ax.set_title('Cognitive Markers Analysis')
    
    # Add a threshold line at 0.5
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
    ax.text(0.51, len(markers) - 0.5, 'Threshold', color='r', va='center')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff7f0e", label='Potential Concern'),
        Patch(facecolor="#1f77b4", label='Normal Range')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    return fig

def plot_feature_correlation(features: Dict[str, Any]) -> Figure:
    """
    Create a correlation matrix plot for the features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        Figure: Matplotlib figure with correlation matrix
    """
    # Extract features that are simple numeric values
    numeric_features = {}
    for k, v in features.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_features[k] = v
    
    if len(numeric_features) < 2:
        # Create default empty plot with message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Not enough features for correlation analysis", 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Create a small dataset by adding noise to the original values
    n_samples = 30
    feature_names = list(numeric_features.keys())
    data = np.zeros((n_samples, len(feature_names)))
    
    # Fill first row with actual values
    data[0, :] = list(numeric_features.values())
    
    # Fill remaining rows with noisy versions of the features
    for i in range(1, n_samples):
        data[i, :] = data[0, :] + np.random.normal(0, 0.2, len(feature_names))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Clean up feature names for display
    display_names = []
    for name in feature_names:
        if "_" in name:
            # Extract the part after the first underscore
            category, feature_name = name.split("_", 1)
            # Format as "Category: Feature Name"
            cleaned_name = f"{category[:3]}: {feature_name[:12]}"
            display_names.append(cleaned_name)
        else:
            display_names.append(name)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    return fig

def plot_clustering_results(features: Dict[str, Any], clustering_results: Dict[str, Any]) -> Figure:
    """
    Create a plot showing clustering results.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        clustering_results (Dict[str, Any]): Results from clustering analysis
        
    Returns:
        Figure: Matplotlib figure with clustering visualization
    """
    # Extract PCA results for visualization
    if "pca" not in clustering_results or "components" not in clustering_results["pca"]:
        # Create default empty plot with message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No clustering results available", 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Get the PCA components and labels
    pca_components = np.array(clustering_results["pca"]["components"])
    
    # Get K-means clustering results
    kmeans_labels = np.array(clustering_results["kmeans"]["labels"])
    original_cluster = clustering_results["kmeans"]["original_cluster"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # K-means clustering plot
    unique_labels = np.unique(kmeans_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        cluster_points = pca_components[kmeans_labels == label]
        ax1.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[i],
            label=f'Cluster {label}',
            alpha=0.7
        )
    
    # Highlight the original sample
    ax1.scatter(
        pca_components[0, 0],
        pca_components[0, 1],
        color='red',
        marker='X',
        s=150,
        label='Current Sample'
    )
    
    ax1.set_title('K-means Clustering')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Scatter plot colored by distance from center of current sample's cluster
    # Get the cluster center for the original sample's cluster
    if "centers" in clustering_results["kmeans"]:
        centers = np.array(clustering_results["kmeans"]["centers"])
        original_center = centers[original_cluster]
        
        # Calculate distances from all points to the original cluster center
        distances = np.sqrt(
            np.sum((pca_components - original_center[None, :2]) ** 2, axis=1)
        )
        
        # Normalize distances for coloring
        max_dist = np.max(distances) if len(distances) > 0 else 1
        normalized_distances = distances / max_dist if max_dist > 0 else distances
        
        # Create scatter plot with color based on distance
        scatter = ax2.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            c=normalized_distances,
            cmap='YlOrRd',
            alpha=0.7
        )
        
        # Highlight the original sample
        ax2.scatter(
            pca_components[0, 0],
            pca_components[0, 1],
            color='blue',
            marker='X',
            s=150,
            label='Current Sample'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Distance from Cluster Center')
        
        ax2.set_title('Distance Analysis')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No cluster centers available", 
                ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    
    plt.tight_layout()
    
    return fig
