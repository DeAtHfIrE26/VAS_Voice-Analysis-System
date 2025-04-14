import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Any
import datetime

def _figure_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: matplotlib figure
        
    Returns:
        str: base64 encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def _generate_metrics_table(features: Dict[str, Any]) -> str:
    """
    Generate an HTML table with key metrics.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        
    Returns:
        str: HTML table with metrics
    """
    # Define key metrics to display
    key_metrics = {
        "Linguistic Features": [
            ("Hesitation Frequency", features.get("linguistic_hesitation_frequency", 0)),
            ("Type-Token Ratio", features.get("linguistic_type_token_ratio", 0)),
            ("Word Retrieval Issues", features.get("linguistic_word_retrieval_issues", 0))
        ],
        "Acoustic Features": [
            ("Pause Rate", features.get("acoustic_pause_rate", 0)),
            ("Harmonics-to-Noise Ratio", features.get("acoustic_harmonics_to_noise_ratio", 0)),
            ("Jitter", features.get("acoustic_jitter", 0))
        ],
        "Temporal Features": [
            ("Speech Rate (WPM)", features.get("temporal_speech_rate_wpm", 0)),
            ("Response Latency", features.get("temporal_avg_response_latency", 0)),
            ("Long Pause Rate", features.get("temporal_long_pause_rate", 0))
        ]
    }
    
    # Generate HTML table
    html = '<table class="metrics-table">'
    
    # Table header
    html += '<thead><tr>'
    for category in key_metrics.keys():
        html += f'<th>{category}</th>'
    html += '</tr></thead>'
    
    # Table body
    html += '<tbody>'
    
    # Find the maximum number of metrics in any category
    max_metrics = max(len(metrics) for metrics in key_metrics.values())
    
    # Generate rows
    for i in range(max_metrics):
        html += '<tr>'
        for category, metrics in key_metrics.items():
            if i < len(metrics):
                metric_name, metric_value = metrics[i]
                
                # Format the value based on its magnitude
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.3f}"
                else:
                    formatted_value = str(metric_value)
                
                html += f'<td><span class="metric-name">{metric_name}:</span> {formatted_value}</td>'
            else:
                html += '<td></td>'
        html += '</tr>'
    
    html += '</tbody></table>'
    
    return html

def _generate_risk_assessment(analysis_results: Dict[str, Any]) -> str:
    """
    Generate risk assessment HTML based on analysis results.
    
    Args:
        analysis_results (Dict[str, Any]): Machine learning analysis results
        
    Returns:
        str: HTML with risk assessment
    """
    html = '<div class="risk-assessment">'
    
    if "anomalies" in analysis_results and "risk_score" in analysis_results["anomalies"]:
        risk_score = analysis_results["anomalies"]["risk_score"]
        
        # Determine risk level
        if risk_score < 0.33:
            risk_level = "Low"
            risk_color = "green"
        elif risk_score < 0.66:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        # Create risk meter visualization
        html += '<h3>Cognitive Risk Assessment</h3>'
        html += f'<div class="risk-meter">'
        html += f'<div class="risk-label">Risk Level: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></div>'
        html += f'<div class="risk-bar-container">'
        html += f'<div class="risk-bar" style="width: {risk_score * 100}%; background-color: {risk_color};"></div>'
        html += f'</div>'
        html += f'<div class="risk-score">Score: {risk_score:.2f}</div>'
        html += '</div>'
        
        # Add significant markers if available
        if "significant_markers" in analysis_results["anomalies"] and analysis_results["anomalies"]["significant_markers"]:
            significant_markers = analysis_results["anomalies"]["significant_markers"]
            
            html += '<div class="significant-markers">'
            html += '<h4>Potential Areas of Concern</h4>'
            html += '<ul>'
            
            for marker_info in significant_markers:
                marker = marker_info["marker"]
                html += f'<li><strong>{marker}</strong>'
                
                # Add feature contributions if available
                if "feature_contributions" in marker_info:
                    top_features = list(marker_info["feature_contributions"].items())[:3]
                    if top_features:
                        html += '<ul class="feature-list">'
                        for feature, contribution in top_features:
                            # Clean up feature name
                            display_name = feature.split("_", 1)[1].replace("_", " ").title()
                            html += f'<li>{display_name}</li>'
                        html += '</ul>'
                
                html += '</li>'
            
            html += '</ul>'
            html += '</div>'
    else:
        html += '<p>Insufficient data for risk assessment.</p>'
    
    html += '</div>'
    
    return html

def _generate_feature_insights(features: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
    """
    Generate insights about key features.
    
    Args:
        features (Dict[str, Any]): Dictionary of extracted features
        analysis_results (Dict[str, Any]): Machine learning analysis results
        
    Returns:
        str: HTML with feature insights
    """
    html = '<div class="feature-insights">'
    html += '<h3>Feature Insights</h3>'
    
    # Check if feature importance is available
    if "importance" in analysis_results and "top_features" in analysis_results["importance"]:
        top_features = analysis_results["importance"]["top_features"]
        
        html += '<div class="top-features">'
        html += '<h4>Most Important Features</h4>'
        html += '<ul>'
        
        for feature, importance in list(top_features.items())[:5]:
            # Clean up feature name
            if "_" in feature:
                category, feature_name = feature.split("_", 1)
                display_name = f"{category.capitalize()}: {feature_name.replace('_', ' ').title()}"
            else:
                display_name = feature
            
            html += f'<li><strong>{display_name}</strong> (Importance: {importance:.3f})</li>'
        
        html += '</ul>'
        html += '</div>'
        
        # Add category importance if available
        if "categories" in analysis_results["importance"]:
            categories = analysis_results["importance"]["categories"]
            
            html += '<div class="category-importance">'
            html += '<h4>Feature Category Importance</h4>'
            
            # Create a simple bar chart
            html += '<div class="category-bars">'
            
            for category, importance in categories.items():
                # Determine bar color
                if category == "linguistic":
                    color = "#1f77b4"  # Blue
                elif category == "acoustic":
                    color = "#ff7f0e"  # Orange
                elif category == "temporal":
                    color = "#2ca02c"  # Green
                else:
                    color = "#7f7f7f"  # Gray
                
                html += f'<div class="category-bar-container">'
                html += f'<div class="category-label">{category.capitalize()}</div>'
                html += f'<div class="category-bar-wrapper">'
                html += f'<div class="category-bar" style="width: {importance * 100}%; background-color: {color};"></div>'
                html += f'</div>'
                html += f'<div class="category-value">{importance:.3f}</div>'
                html += '</div>'
            
            html += '</div>'
            html += '</div>'
    else:
        html += '<p>Feature importance analysis not available.</p>'
    
    html += '</div>'
    
    return html

def generate_report(
    audio_info: Dict[str, Any],
    transcription: Dict[str, Any],
    features: Dict[str, Any],
    analysis_results: Dict[str, Any]
) -> str:
    """
    Generate a comprehensive HTML report.
    
    Args:
        audio_info (Dict[str, Any]): Information about the audio file
        transcription (Dict[str, Any]): Transcription results
        features (Dict[str, Any]): Extracted features
        analysis_results (Dict[str, Any]): Machine learning analysis results
        
    Returns:
        str: Complete HTML report
    """
    # Generate figures for the report
    from modules.visualization import (
        plot_feature_distributions, 
        plot_anomaly_detection,
        plot_feature_correlation
    )
    
    # Generate figures
    feature_dist_fig = plot_feature_distributions(features)
    feature_dist_img = _figure_to_base64(feature_dist_fig)
    plt.close(feature_dist_fig)
    
    if "anomalies" in analysis_results:
        anomaly_fig = plot_anomaly_detection(features, analysis_results["anomalies"])
        anomaly_img = _figure_to_base64(anomaly_fig)
        plt.close(anomaly_fig)
    else:
        anomaly_img = None
    
    correlation_fig = plot_feature_correlation(features)
    correlation_img = _figure_to_base64(correlation_fig)
    plt.close(correlation_fig)
    
    # Current date for the report
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Start building the HTML report
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cognitive Assessment Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 20px;
                background-color: #f9f9f9;
            }
            .report-container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }
            .report-header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .report-header h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .report-header p {
                color: #7f8c8d;
                margin: 5px 0;
            }
            .report-section {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .report-section:last-child {
                border-bottom: none;
            }
            .report-section h2 {
                color: #2c3e50;
                margin-bottom: 15px;
            }
            .audio-info {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .audio-info-item {
                text-align: center;
                padding: 10px;
                flex: 1;
                min-width: 120px;
            }
            .audio-info-item .label {
                font-weight: bold;
                margin-bottom: 5px;
                color: #7f8c8d;
            }
            .audio-info-item .value {
                font-size: 1.2em;
                color: #2c3e50;
            }
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .metrics-table th {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                text-align: left;
            }
            .metrics-table td {
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            .metrics-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .metric-name {
                font-weight: bold;
                color: #2c3e50;
            }
            .transcription {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                white-space: pre-wrap;
                margin-bottom: 20px;
                line-height: 1.8;
            }
            .visualization {
                text-align: center;
                margin: 20px 0;
            }
            .visualization img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }
            .risk-assessment {
                margin: 20px 0;
            }
            .risk-meter {
                margin: 20px 0;
            }
            .risk-label {
                margin-bottom: 5px;
                font-size: 1.1em;
            }
            .risk-bar-container {
                height: 25px;
                background-color: #eee;
                border-radius: 15px;
                margin: 10px 0;
                position: relative;
            }
            .risk-bar {
                height: 100%;
                border-radius: 15px;
                transition: width 0.5s ease-in-out;
            }
            .risk-score {
                text-align: right;
                font-weight: bold;
            }
            .significant-markers {
                margin-top: 20px;
            }
            .feature-list {
                margin-top: 5px;
                padding-left: 20px;
            }
            .feature-insights {
                margin: 20px 0;
            }
            .category-bars {
                margin-top: 15px;
            }
            .category-bar-container {
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            }
            .category-label {
                width: 100px;
                font-weight: bold;
            }
            .category-bar-wrapper {
                flex-grow: 1;
                height: 20px;
                background-color: #eee;
                border-radius: 10px;
                margin: 0 10px;
            }
            .category-bar {
                height: 100%;
                border-radius: 10px;
            }
            .category-value {
                width: 50px;
                text-align: right;
            }
            .footer {
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="report-container">
    """
    
    # Report header
    html += f"""
            <div class="report-header">
                <h1>Cognitive Assessment Report</h1>
                <p>Generated on {current_date}</p>
                <p>MemoTag Voice Analysis System</p>
            </div>
    """
    
    # Audio information section
    html += f"""
            <div class="report-section">
                <h2>Audio Information</h2>
                <div class="audio-info">
                    <div class="audio-info-item">
                        <div class="label">Duration</div>
                        <div class="value">{audio_info.get('duration', 0):.2f}s</div>
                    </div>
                    <div class="audio-info-item">
                        <div class="label">Sample Rate</div>
                        <div class="value">{audio_info.get('sample_rate', 0)} Hz</div>
                    </div>
                    <div class="audio-info-item">
                        <div class="label">File Format</div>
                        <div class="value">{audio_info.get('format', 'Unknown')}</div>
                    </div>
                </div>
            </div>
    """
    
    # Transcription section
    html += f"""
            <div class="report-section">
                <h2>Speech Transcription</h2>
                <div class="transcription">
                    {transcription.get('text', 'No transcription available.')}
                </div>
            </div>
    """
    
    # Key metrics section
    html += f"""
            <div class="report-section">
                <h2>Key Speech Metrics</h2>
                {_generate_metrics_table(features)}
            </div>
    """
    
    # Visualization section
    html += f"""
            <div class="report-section">
                <h2>Visualizations</h2>
                
                <div class="visualization">
                    <h3>Feature Distribution</h3>
                    <img src="data:image/png;base64,{feature_dist_img}" alt="Feature Distribution">
                </div>
    """
    
    if anomaly_img:
        html += f"""
                <div class="visualization">
                    <h3>Cognitive Markers Analysis</h3>
                    <img src="data:image/png;base64,{anomaly_img}" alt="Anomaly Detection">
                </div>
        """
    
    html += f"""
                <div class="visualization">
                    <h3>Feature Correlation</h3>
                    <img src="data:image/png;base64,{correlation_img}" alt="Feature Correlation">
                </div>
            </div>
    """
    
    # Assessment and Insights section
    html += f"""
            <div class="report-section">
                <h2>Assessment and Insights</h2>
                
                {_generate_risk_assessment(analysis_results)}
                
                {_generate_feature_insights(features, analysis_results)}
            </div>
    """
    
    # Footer
    html += f"""
            <div class="footer">
                <p>This report is generated for research purposes only and should not be used for medical diagnosis.</p>
                <p>© {datetime.datetime.now().year} MemoTag Voice Analysis System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html
