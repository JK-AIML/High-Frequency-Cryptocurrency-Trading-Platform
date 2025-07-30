"""
Drift Detection Module

This module provides functionality to detect data drift between two datasets.
It includes various statistical tests and metrics to measure drift in data distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of data drift that can be detected."""
    COVARIATE = auto()       # Drift in feature distributions
    CONCEPT = auto()         # Drift in target concept
    LABEL = auto()           # Drift in label distribution
    PRIOR_PROBABILITY = auto() # Drift in class priors
    
class DriftMetric(Enum):
    """Available drift detection metrics."""
    KOLMOGOROV_SMIRNOV = "ks_test"
    JENSEN_SHANNON = "js_divergence"
    WASSERSTEIN = "wasserstein"
    CHI_SQUARED = "chi_squared"
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    MAHALANOBIS = "mahalanobis"
    ISOLATION_FOREST = "isolation_forest"
    
@dataclass
class DriftResult:
    """Container for drift detection results."""
    metric: DriftMetric
    statistic: float
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    is_drifted: bool = False
    feature: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'metric': self.metric.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'threshold': self.threshold,
            'is_drifted': self.is_drifted,
            'feature': self.feature,
            'message': self.message
        }

class DriftDetector:
    """
    Detects drift between reference and current data distributions.
    
    This class provides methods to detect various types of data drift using
    statistical tests and distance metrics.
    """
    
    def __init__(
        self,
        reference_data: Union[pd.DataFrame, np.ndarray],
        current_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference dataset (baseline)
            current_data: Current dataset to compare against reference
            feature_names: Optional list of feature names
            alpha: Significance level for statistical tests
            random_state: Random seed for reproducibility
        """
        self.reference_data = self._ensure_dataframe(reference_data, feature_names)
        self.current_data = self._ensure_dataframe(current_data, feature_names)
        self.alpha = alpha
        self.random_state = random_state
        self._validate_inputs()
    
    def _ensure_dataframe(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert input to pandas DataFrame if needed."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=feature_names)
        else:
            raise ValueError("Input data must be pandas DataFrame or numpy array")
    
    def _validate_inputs(self) -> None:
        """Validate input datasets."""
        if self.reference_data.shape[1] != self.current_data.shape[1]:
            raise ValueError("Reference and current data must have the same number of features")
        
        if len(self.reference_data) == 0 or len(self.current_data) == 0:
            raise ValueError("Input datasets cannot be empty")
    
    def detect_drift(
        self,
        method: Union[DriftMetric, str] = DriftMetric.KOLMOGOROV_SMIRNOV,
        features: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, DriftResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            method: Drift detection method to use
            features: List of features to analyze (default: all features)
            **kwargs: Additional arguments for the drift detection method
            
        Returns:
            Dictionary mapping feature names to drift results
        """
        if isinstance(method, str):
            method = DriftMetric(method.lower())
        
        if features is None:
            features = self.reference_data.columns.tolist()
        
        results = {}
        
        for feature in features:
            ref = self.reference_data[feature].dropna()
            curr = self.current_data[feature].dropna()
            
            if method == DriftMetric.KOLMOGOROV_SMIRNOV:
                result = self._ks_test(ref, curr, **kwargs)
            elif method == DriftMetric.JENSEN_SHANNON:
                result = self._js_divergence(ref, curr, **kwargs)
            elif method == DriftMetric.WASSERSTEIN:
                result = self._wasserstein_distance(ref, curr, **kwargs)
            elif method == DriftMetric.CHI_SQUARED:
                result = self._chi_squared_test(ref, curr, **kwargs)
            elif method == DriftMetric.PSI:
                result = self._psi_metric(ref, curr, **kwargs)
            elif method == DriftMetric.KL_DIVERGENCE:
                result = self._kl_divergence(ref, curr, **kwargs)
            elif method == DriftMetric.MAHALANOBIS:
                result = self._mahalanobis_distance(ref, curr, **kwargs)
            elif method == DriftMetric.ISOLATION_FOREST:
                result = self._isolation_forest(ref, curr, **kwargs)
            else:
                raise ValueError(f"Unsupported drift detection method: {method}")
            
            result.feature = feature
            results[feature] = result
        
        return results
    
    def _ks_test(
        self,
        ref: pd.Series,
        curr: pd.Series,
        alpha: Optional[float] = None
    ) -> DriftResult:
        """Kolmogorov-Smirnov test for distribution similarity."""
        alpha = alpha or self.alpha
        stat, p_value = ks_2samp(ref, curr)
        
        return DriftResult(
            metric=DriftMetric.KOLMOGOROV_SMIRNOV,
            statistic=stat,
            p_value=p_value,
            threshold=alpha,
            is_drifted=p_value < alpha,
            message=f"KS test {'detected' if p_value < alpha else 'did not detect'} drift"
        )
    
    def _js_divergence(
        self,
        ref: pd.Series,
        curr: pd.Series,
        threshold: float = 0.1,
        bins: int = 10,
        **kwargs
    ) -> DriftResult:
        """Jensen-Shannon divergence between distributions."""
        # Create histograms with the same bins
        min_val = min(ref.min(), curr.min())
        max_val = max(ref.max(), curr.max())
        
        # Handle constant values
        if max_val == min_val:
            return DriftResult(
                metric=DriftMetric.JENSEN_SHANNON,
                statistic=0.0,
                threshold=threshold,
                is_drifted=False,
                message="Constant values detected, no divergence"
            )
            
        hist_ref = np.histogram(ref, bins=bins, range=(min_val, max_val), density=True)[0]
        hist_curr = np.histogram(curr, bins=bins, range=(min_val, max_val), density=True)[0]
        
        # Add small constant to avoid division by zero
        hist_ref = hist_ref + 1e-10
        hist_curr = hist_curr + 1e-10
        
        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_curr = hist_curr / hist_curr.sum()
        
        # Calculate JS divergence
        m = 0.5 * (hist_ref + hist_curr)
        js = 0.5 * (stats.entropy(hist_ref, m) + stats.entropy(hist_curr, m))
        js = np.sqrt(js)  # Take square root to get JS distance
        
        return DriftResult(
            metric=DriftMetric.JENSEN_SHANNON,
            statistic=js,
            threshold=threshold,
            is_drifted=js > threshold,
            message=f"JS divergence {js:.4f} is {'>' if js > threshold else '<='} threshold {threshold}"
        )
    
    def _wasserstein_distance(
        self,
        ref: pd.Series,
        curr: pd.Series,
        threshold: float = 0.1,
        **kwargs
    ) -> DriftResult:
        """Wasserstein distance between distributions."""
        dist = wasserstein_distance(ref, curr)
        
        return DriftResult(
            metric=DriftMetric.WASSERSTEIN,
            statistic=dist,
            threshold=threshold,
            is_drifted=dist > threshold,
            message=f"Wasserstein distance {dist:.4f} is {'>' if dist > threshold else '<='} threshold {threshold}"
        )
    
    def _chi_squared_test(
        self,
        ref: pd.Series,
        curr: pd.Series,
        alpha: Optional[float] = None,
        bins: int = 10,
        **kwargs
    ) -> DriftResult:
        """Chi-squared test for categorical distributions."""
        alpha = alpha or self.alpha
        
        # Create bins for continuous data
        if np.issubdtype(ref.dtype, np.number):
            min_val = min(ref.min(), curr.min())
            max_val = max(ref.max(), curr.max())
            bins = np.linspace(min_val, max_val, bins + 1)
            ref_bins = pd.cut(ref, bins=bins, include_lowest=True)
            curr_bins = pd.cut(curr, bins=bins, include_lowest=True)
        else:
            ref_bins = ref
            curr_bins = curr
        
        # Create contingency table
        ref_counts = ref_bins.value_counts().sort_index()
        curr_counts = curr_bins.value_counts().sort_index()
        
        # Align indices
        all_categories = ref_counts.index.union(curr_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        curr_counts = curr_counts.reindex(all_categories, fill_value=0)
        
        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency([ref_counts, curr_counts])
        
        return DriftResult(
            metric=DriftMetric.CHI_SQUARED,
            statistic=chi2,
            p_value=p_value,
            threshold=alpha,
            is_drifted=p_value < alpha,
            message=f"Chi-squared test {'detected' if p_value < alpha else 'did not detect'} drift"
        )
    
    def _psi_metric(
        self,
        ref: pd.Series,
        curr: pd.Series,
        threshold: float = 0.1,
        bins: int = 10,
        **kwargs
    ) -> DriftResult:
        """Population Stability Index (PSI) metric."""
        # Create bins
        min_val = min(ref.min(), curr.min())
        max_val = max(ref.max(), curr.max())
        
        # Handle constant values
        if max_val == min_val:
            return DriftResult(
                metric=DriftMetric.PSI,
                statistic=0.0,
                threshold=threshold,
                is_drifted=False,
                message="Constant values detected, PSI is 0"
            )
        
        bins = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate histograms
        hist_ref = np.histogram(ref, bins=bins)[0]
        hist_curr = np.histogram(curr, bins=bins)[0]
        
        # Add small constant to avoid division by zero
        hist_ref = hist_ref + 1e-10
        hist_curr = hist_curr + 1e-10
        
        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_curr = hist_curr / hist_curr.sum()
        
        # Calculate PSI
        psi = np.sum((hist_curr - hist_ref) * np.log(hist_curr / hist_ref))
        
        # Interpretation
        if psi < 0.1:
            message = "No significant population change"
        elif psi < 0.2:
            message = "Moderate population change"
        else:
            message = "Significant population change"
        
        return DriftResult(
            metric=DriftMetric.PSI,
            statistic=psi,
            threshold=threshold,
            is_drifted=psi > threshold,
            message=f"PSI: {psi:.4f} - {message}"
        )
    
    def _kl_divergence(
        self,
        ref: pd.Series,
        curr: pd.Series,
        threshold: float = 0.1,
        bins: int = 10,
        **kwargs
    ) -> DriftResult:
        """Kullback-Leibler divergence between distributions."""
        # Create histograms with the same bins
        min_val = min(ref.min(), curr.min())
        max_val = max(ref.max(), curr.max())
        
        # Handle constant values
        if max_val == min_val:
            return DriftResult(
                metric=DriftMetric.KL_DIVERGENCE,
                statistic=0.0,
                threshold=threshold,
                is_drifted=False,
                message="Constant values detected, KL divergence is 0"
            )
        
        hist_ref = np.histogram(ref, bins=bins, range=(min_val, max_val), density=True)[0]
        hist_curr = np.histogram(curr, bins=bins, range=(min_val, max_val), density=True)[0]
        
        # Add small constant to avoid division by zero
        hist_ref = hist_ref + 1e-10
        hist_curr = hist_curr + 1e-10
        
        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_curr = hist_curr / hist_curr.sum()
        
        # Calculate KL divergence
        kl = stats.entropy(hist_ref, hist_curr)
        
        return DriftResult(
            metric=DriftMetric.KL_DIVERGENCE,
            statistic=kl,
            threshold=threshold,
            is_drifted=kl > threshold,
            message=f"KL divergence {kl:.4f} is {'>' if kl > threshold else '<='} threshold {threshold}"
        )
    
    def _mahalanobis_distance(
        self,
        ref: pd.Series,
        curr: pd.Series,
        threshold: float = 3.0,
        **kwargs
    ) -> DriftResult:
        """Mahalanobis distance for multivariate outlier detection."""
        # Reshape for sklearn
        X_ref = ref.values.reshape(-1, 1)
        X_curr = curr.values.reshape(-1, 1)
        
        # Fit robust covariance estimator
        cov = EllipticEnvelope(contamination=0.1, random_state=self.random_state)
        cov.fit(X_ref)
        
        # Calculate Mahalanobis distance for current data
        dist = cov.mahalanobis(X_curr)
        mean_dist = np.mean(dist)
        
        return DriftResult(
            metric=DriftMetric.MAHALANOBIS,
            statistic=mean_dist,
            threshold=threshold,
            is_drifted=mean_dist > threshold,
            message=f"Mean Mahalanobis distance {mean_dist:.4f} is {'>' if mean_dist > threshold else '<='} threshold {threshold}"
        )
    
    def _isolation_forest(
        self,
        ref: pd.Series,
        curr: pd.Series,
        contamination: float = 0.1,
        threshold: float = 0.5,
        **kwargs
    ) -> DriftResult:
        """Isolation Forest for anomaly detection."""
        # Reshape for sklearn
        X_ref = ref.values.reshape(-1, 1)
        X_curr = curr.values.reshape(-1, 1)
        
        # Train isolation forest
        clf = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        clf.fit(X_ref)
        
        # Predict anomalies in current data
        pred = clf.predict(X_curr)
        anomaly_ratio = (pred == -1).mean()
        
        return DriftResult(
            metric=DriftMetric.ISOLATION_FOREST,
            statistic=anomaly_ratio,
            threshold=threshold,
            is_drifted=anomaly_ratio > threshold,
            message=f"Anomaly ratio {anomaly_ratio:.4f} is {'>' if anomaly_ratio > threshold else '<='} threshold {threshold}"
        )
    
    def plot_distributions(
        self,
        feature: str,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Plot distributions of reference and current data for a feature."""
        plt.figure(figsize=figsize)
        
        # Plot histograms
        sns.histplot(
            self.reference_data[feature], 
            color="blue", 
            label="Reference", 
            alpha=0.5,
            kde=True,
            stat="density",
            **kwargs
        )
        
        sns.histplot(
            self.current_data[feature],
            color="red",
            label="Current",
            alpha=0.5,
            kde=True,
            stat="density",
            **kwargs
        )
        
        # Add labels and title
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.title(title or f"Distribution of {feature}")
        plt.legend()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def generate_report(
        self,
        methods: Optional[List[Union[DriftMetric, str]]] = None,
        features: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift detection report.
        
        Args:
            methods: List of drift detection methods to use
            features: List of features to analyze
            output_file: Path to save the report (HTML)
            **kwargs: Additional arguments for drift detection methods
            
        Returns:
            Dictionary containing the report data
        """
        if methods is None:
            methods = [m for m in DriftMetric]
        
        if features is None:
            features = self.reference_data.columns.tolist()
        
        report = {
            'summary': {
                'n_features': len(features),
                'n_samples_ref': len(self.reference_data),
                'n_samples_curr': len(self.current_data),
                'drift_detected': False,
                'drift_summary': {}
            },
            'features': {},
            'metrics': {}
        }
        
        # Run all specified drift detection methods
        for method in methods:
            results = self.detect_drift(method=method, features=features, **kwargs)
            
            # Update feature-level results
            for feature, result in results.items():
                if feature not in report['features']:
                    report['features'][feature] = {}
                report['features'][feature][method.value] = result.to_dict()
                
                # Update drift summary
                if result.is_drifted:
                    if method.value not in report['summary']['drift_summary']:
                        report['summary']['drift_summary'][method.value] = []
                    report['summary']['drift_summary'][method.value].append(feature)
                    report['summary']['drift_detected'] = True
            
            # Update metrics summary
            if method.value not in report['metrics']:
                report['metrics'][method.value] = {
                    'drift_count': 0,
                    'total_tests': len(results),
                    'drift_ratio': 0.0,
                    'features_with_drift': []
                }
            
            drift_count = sum(1 for r in results.values() if r.is_drifted)
            report['metrics'][method.value].update({
                'drift_count': drift_count,
                'drift_ratio': drift_count / len(results) if results else 0.0,
                'features_with_drift': [f for f, r in results.items() if r.is_drifted]
            })
        
        # Generate HTML report if output file is specified
        if output_file:
            self._generate_html_report(report, output_file)
        
        return report
    
    def _generate_html_report(
        self,
        report: Dict[str, Any],
        output_file: str
    ) -> None:
        """Generate an HTML report from the drift analysis."""
        # Import here to avoid dependency if not generating HTML
        from jinja2 import Environment, FileSystemLoader
        import os
        import json
        
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, 'templates')
        
        # Create templates directory if it doesn't exist
        os.makedirs(templates_dir, exist_ok=True)
        
        # Create a simple HTML template
        template_path = os.path.join(templates_dir, 'drift_report_template.html')
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Drift Detection Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        .header { background-color: #f4f4f4; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        .drift-detected { border-left: 5px solid #ff6b6b; }
        .no-drift { border-left: 5px solid #51cf66; }
        .feature-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .feature-table th, .feature-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .feature-table th { background-color: #f2f2f2; }
        .feature-table tr:nth-child(even) { background-color: #f9f9f9; }
        .badge {
            display: inline-block;
            padding: 3px 7px;
            font-size: 12px;
            font-weight: bold;
            line-height: 1;
            color: #fff;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 10px;
        }
        .badge-danger { background-color: #dc3545; }
        .badge-success { background-color: #28a745; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Drift Detection Report</h1>
        <p>Generated on: {{ generated_at }}</p>
        <p>Reference samples: {{ report.summary.n_samples_ref }}</p>
        <p>Current samples: {{ report.summary.n_samples_curr }}</p>
        <p>Features analyzed: {{ report.summary.n_features }}</p>
        <p>Drift detected: 
            {% if report.summary.drift_detected %}
                <span class="badge badge-danger">Yes</span>
            {% else %}
                <span class="badge badge-success">No</span>
            {% endif %}
        </p>
    </div>

    <div class="section">
        <h2>Metrics Summary</h2>
        {% for metric, data in report.metrics.items() %}
        <div class="metric-card {% if data.drift_count > 0 %}drift-detected{% else %}no-drift{% end %}">
            <h3>{{ metric|title }}</h3>
            <p>Drift detected in {{ data.drift_count }} of {{ data.total_tests }} features ({{ '%.1f'|format(data.drift_ratio * 100) }}%)</p>
            {% if data.drift_count > 0 %}
                <p>Features with drift: {{ ', '.join(data.features_with_drift) }}</p>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Feature Details</h2>
        {% for feature, metrics in report.features.items() %}
        <div class="metric-card">
            <h3>{{ feature }}</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Statistic</th>
                        <th>P-value</th>
                        <th>Threshold</th>
                        <th>Drift Detected</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, result in metrics.items() %}
                    <tr>
                        <td>{{ result.metric }}</td>
                        <td>{{ '%.4f'|format(result.statistic) if result.statistic is not none else 'N/A' }}</td>
                        <td>{{ '%.4f'|format(result.p_value) if result.p_value is not none else 'N/A' }}</td>
                        <td>{{ '%.4f'|format(result.threshold) if result.threshold is not none else 'N/A' }}</td>
                        <td>
                            {% if result.is_drifted %}
                                <span class="badge badge-danger">Yes</span>
                            {% else %}
                                <span class="badge badge-success">No</span>
                            {% endif %}
                        </td>
                        <td>{{ result.message }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endfor %}
    </div>
</body>
</html>
""")
        
        # Set up Jinja2 environment
        env = Environment(loader=FileSystemLoader(templates_dir))
        template = env.get_template('drift_report_template.html')
        
        # Add custom filters
        template.globals.update({
            'zip': zip,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'round': round,
        })
        
        # Add current timestamp
        report['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Render the template with report data
        html = template.render(report=report)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report generated at: {output_file}")

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Reference data (normal distribution)
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.exponential(1, n_samples)
    })
    
    # Current data (shifted distributions)
    curr_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),  # Slight shift in mean and std
        'feature2': np.random.normal(7, 2.5, n_samples),    # Shifted mean and increased std
        'feature3': np.random.exponential(1.5, n_samples)   # Different scale
    })
    
    # Initialize drift detector
    detector = DriftDetector(ref_data, curr_data)
    
    # Detect drift using multiple methods
    methods = [
        DriftMetric.KOLMOGOROV_SMIRNOV,
        DriftMetric.JENSEN_SHANNON,
        DriftMetric.WASSERSTEIN,
        DriftMetric.PSI
    ]
    
    # Generate report
    report = detector.generate_report(
        methods=methods,
        output_file="drift_report.html"
    )
    
    # Plot distributions for features with detected drift
    for feature in ref_data.columns:
        if any(report['features'][feature][m.value]['is_drifted'] for m in methods):
            detector.plot_distributions(
                feature=feature,
                title=f"Distribution of {feature} - Drift Detected"
            )
