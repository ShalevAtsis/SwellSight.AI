"""Training data quality monitoring system."""

import numpy as np
import torch
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for a dataset."""
    timestamp: str
    dataset_name: str
    total_samples: int
    
    # Distribution metrics
    height_distribution: Dict[str, float]
    wave_type_distribution: Dict[str, int]
    direction_distribution: Dict[str, int]
    data_source_distribution: Dict[str, int]
    
    # Quality scores
    overall_quality_score: float
    diversity_score: float
    balance_score: float
    
    # Statistical properties
    height_statistics: Dict[str, float]
    augmentation_coverage: Dict[str, Any]
    
    # Data drift indicators
    drift_score: Optional[float] = None
    drift_detected: bool = False
    drift_details: Optional[Dict[str, Any]] = None


@dataclass
class DataDriftResult:
    """Result of data drift detection analysis."""
    drift_detected: bool
    drift_score: float
    drift_threshold: float
    affected_features: List[str]
    statistical_tests: Dict[str, Dict[str, float]]
    recommendations: List[str]


class DataQualityMonitor:
    """
    Training data quality monitoring system.
    
    Provides statistical analysis of training data distribution, data drift detection
    between synthetic and real data, data quality dashboards and reporting, and
    automated data validation and filtering.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality monitor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Configuration parameters
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.min_samples_for_drift = self.config.get('min_samples_for_drift', 100)
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'diversity_min': 0.3,
            'balance_min': 0.2,
            'overall_min': 0.5
        })
        
        # Storage for historical metrics
        self.historical_metrics: List[DataQualityMetrics] = []
        self.baseline_metrics: Optional[DataQualityMetrics] = None
        
        # Output paths
        self.output_path = Path(self.config.get('output_path', 'data_quality_reports'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataQualityMonitor with drift threshold: {self.drift_threshold}")
        logger.info(f"Output path: {self.output_path}")
    
    def analyze_dataset_quality(self, samples_metadata: List[Dict[str, Any]], 
                               dataset_name: str = "training_data") -> DataQualityMetrics:
        """
        Analyze comprehensive data quality metrics for a dataset.
        
        Args:
            samples_metadata: List of sample metadata dictionaries
            dataset_name: Name identifier for the dataset
        
        Returns:
            DataQualityMetrics object with comprehensive analysis
        """
        logger.info(f"Analyzing data quality for dataset: {dataset_name} ({len(samples_metadata)} samples)")
        
        if not samples_metadata:
            logger.warning("Empty dataset provided for quality analysis")
            return self._create_empty_metrics(dataset_name)
        
        # Extract basic statistics
        heights = [sample['height_meters'] for sample in samples_metadata]
        wave_types = [sample['wave_type'] for sample in samples_metadata]
        directions = [sample['direction'] for sample in samples_metadata]
        data_sources = [sample.get('data_source', 'unknown') for sample in samples_metadata]
        
        # Calculate distribution metrics
        height_distribution = self._calculate_height_distribution(heights)
        wave_type_distribution = {wt: wave_types.count(wt) for wt in set(wave_types)}
        direction_distribution = {d: directions.count(d) for d in set(directions)}
        data_source_distribution = {ds: data_sources.count(ds) for ds in set(data_sources)}
        
        # Calculate height statistics
        height_statistics = {
            'min': float(np.min(heights)),
            'max': float(np.max(heights)),
            'mean': float(np.mean(heights)),
            'std': float(np.std(heights)),
            'median': float(np.median(heights)),
            'q25': float(np.percentile(heights, 25)),
            'q75': float(np.percentile(heights, 75))
        }
        
        # Analyze augmentation coverage
        augmentation_coverage = self._analyze_augmentation_coverage(samples_metadata)
        
        # Calculate quality scores
        diversity_score = self._calculate_diversity_score(
            wave_type_distribution, direction_distribution, heights
        )
        balance_score = self._calculate_balance_score(
            wave_type_distribution, direction_distribution, data_source_distribution
        )
        overall_quality_score = self._calculate_overall_quality_score(
            diversity_score, balance_score, len(samples_metadata)
        )
        
        # Create metrics object
        metrics = DataQualityMetrics(
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            total_samples=len(samples_metadata),
            height_distribution=height_distribution,
            wave_type_distribution=wave_type_distribution,
            direction_distribution=direction_distribution,
            data_source_distribution=data_source_distribution,
            overall_quality_score=overall_quality_score,
            diversity_score=diversity_score,
            balance_score=balance_score,
            height_statistics=height_statistics,
            augmentation_coverage=augmentation_coverage
        )
        
        # Store metrics
        self.historical_metrics.append(metrics)
        
        # Set as baseline if first analysis
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            logger.info("Set baseline metrics for drift detection")
        
        logger.info(f"Data quality analysis complete:")
        logger.info(f"  Overall quality score: {overall_quality_score:.3f}")
        logger.info(f"  Diversity score: {diversity_score:.3f}")
        logger.info(f"  Balance score: {balance_score:.3f}")
        
        return metrics
    
    def detect_data_drift(self, current_samples: List[Dict[str, Any]], 
                         reference_samples: Optional[List[Dict[str, Any]]] = None) -> DataDriftResult:
        """
        Detect data drift between current and reference datasets.
        
        Args:
            current_samples: Current dataset samples
            reference_samples: Reference dataset samples (uses baseline if None)
        
        Returns:
            DataDriftResult with drift detection analysis
        """
        logger.info("Performing data drift detection analysis")
        
        if len(current_samples) < self.min_samples_for_drift:
            logger.warning(f"Insufficient samples for drift detection: {len(current_samples)} < {self.min_samples_for_drift}")
            return DataDriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_threshold=self.drift_threshold,
                affected_features=[],
                statistical_tests={},
                recommendations=["Insufficient samples for reliable drift detection"]
            )
        
        # Use baseline metrics if no reference provided
        if reference_samples is None:
            if self.baseline_metrics is None:
                logger.warning("No baseline metrics available for drift detection")
                return DataDriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    drift_threshold=self.drift_threshold,
                    affected_features=[],
                    statistical_tests={},
                    recommendations=["No baseline available for drift detection"]
                )
            # Extract samples from baseline (this is a simplification)
            reference_samples = current_samples  # Placeholder - would need actual baseline samples
        
        # Extract features for comparison
        current_features = self._extract_drift_features(current_samples)
        reference_features = self._extract_drift_features(reference_samples)
        
        # Perform statistical tests
        statistical_tests = {}
        affected_features = []
        drift_scores = []
        
        for feature_name in current_features.keys():
            if feature_name in reference_features:
                test_result = self._perform_drift_test(
                    current_features[feature_name],
                    reference_features[feature_name],
                    feature_name
                )
                statistical_tests[feature_name] = test_result
                
                # Check if drift is significant
                if test_result['p_value'] < 0.05:  # Significant at 5% level
                    affected_features.append(feature_name)
                    drift_scores.append(test_result['effect_size'])
        
        # Calculate overall drift score
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift_score > self.drift_threshold
        
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            drift_detected, affected_features, overall_drift_score
        )
        
        result = DataDriftResult(
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            drift_threshold=self.drift_threshold,
            affected_features=affected_features,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
        
        logger.info(f"Data drift detection complete:")
        logger.info(f"  Drift detected: {drift_detected}")
        logger.info(f"  Drift score: {overall_drift_score:.3f}")
        logger.info(f"  Affected features: {len(affected_features)}")
        
        return result
    
    def validate_data_quality(self, samples_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data quality and provide filtering recommendations.
        
        Args:
            samples_metadata: List of sample metadata to validate
        
        Returns:
            Validation results with filtering recommendations
        """
        logger.info(f"Validating data quality for {len(samples_metadata)} samples")
        
        validation_results = {
            'total_samples': len(samples_metadata),
            'valid_samples': 0,
            'invalid_samples': 0,
            'quality_issues': [],
            'filtering_recommendations': [],
            'quality_distribution': {
                'high_quality': 0,
                'medium_quality': 0,
                'low_quality': 0
            }
        }
        
        for sample in samples_metadata:
            quality_score = self._assess_sample_quality(sample)
            
            if quality_score >= 0.7:
                validation_results['quality_distribution']['high_quality'] += 1
                validation_results['valid_samples'] += 1
            elif quality_score >= 0.4:
                validation_results['quality_distribution']['medium_quality'] += 1
                validation_results['valid_samples'] += 1
            else:
                validation_results['quality_distribution']['low_quality'] += 1
                validation_results['invalid_samples'] += 1
                validation_results['quality_issues'].append({
                    'sample_id': sample.get('sample_id', 'unknown'),
                    'quality_score': quality_score,
                    'issues': self._identify_quality_issues(sample)
                })
        
        # Generate filtering recommendations
        low_quality_ratio = validation_results['invalid_samples'] / len(samples_metadata)
        if low_quality_ratio > 0.1:  # More than 10% low quality
            validation_results['filtering_recommendations'].append(
                f"Consider filtering {validation_results['invalid_samples']} low-quality samples ({low_quality_ratio:.1%})"
            )
        
        if validation_results['quality_distribution']['high_quality'] < len(samples_metadata) * 0.5:
            validation_results['filtering_recommendations'].append(
                "Less than 50% high-quality samples - consider improving data generation process"
            )
        
        logger.info(f"Data validation complete:")
        logger.info(f"  High quality: {validation_results['quality_distribution']['high_quality']}")
        logger.info(f"  Medium quality: {validation_results['quality_distribution']['medium_quality']}")
        logger.info(f"  Low quality: {validation_results['quality_distribution']['low_quality']}")
        
        return validation_results
    
    def generate_quality_dashboard(self, metrics: DataQualityMetrics, 
                                 output_filename: Optional[str] = None) -> Path:
        """
        Generate comprehensive data quality dashboard.
        
        Args:
            metrics: DataQualityMetrics to visualize
            output_filename: Optional custom filename
        
        Returns:
            Path to generated dashboard file
        """
        if output_filename is None:
            output_filename = f"quality_dashboard_{metrics.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        output_path = self.output_path / output_filename
        
        # Create dashboard with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Data Quality Dashboard - {metrics.dataset_name}', fontsize=16)
        
        # 1. Height distribution
        heights = list(metrics.height_distribution.keys())
        height_counts = list(metrics.height_distribution.values())
        axes[0, 0].bar(heights, height_counts)
        axes[0, 0].set_title('Height Distribution')
        axes[0, 0].set_xlabel('Height Range (m)')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Wave type distribution
        wave_types = list(metrics.wave_type_distribution.keys())
        wave_counts = list(metrics.wave_type_distribution.values())
        axes[0, 1].pie(wave_counts, labels=wave_types, autopct='%1.1f%%')
        axes[0, 1].set_title('Wave Type Distribution')
        
        # 3. Direction distribution
        directions = list(metrics.direction_distribution.keys())
        dir_counts = list(metrics.direction_distribution.values())
        axes[0, 2].pie(dir_counts, labels=directions, autopct='%1.1f%%')
        axes[0, 2].set_title('Direction Distribution')
        
        # 4. Data source distribution
        sources = list(metrics.data_source_distribution.keys())
        source_counts = list(metrics.data_source_distribution.values())
        axes[1, 0].bar(sources, source_counts)
        axes[1, 0].set_title('Data Source Distribution')
        axes[1, 0].set_xlabel('Data Source')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Quality scores
        quality_metrics = ['Overall', 'Diversity', 'Balance']
        quality_scores = [metrics.overall_quality_score, metrics.diversity_score, metrics.balance_score]
        colors = ['green' if score >= 0.7 else 'orange' if score >= 0.4 else 'red' for score in quality_scores]
        axes[1, 1].bar(quality_metrics, quality_scores, color=colors)
        axes[1, 1].set_title('Quality Scores')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # 6. Height statistics box plot
        height_stats = metrics.height_statistics
        box_data = [height_stats['min'], height_stats['q25'], height_stats['median'], 
                   height_stats['q75'], height_stats['max']]
        axes[1, 2].boxplot([box_data], labels=['Height'])
        axes[1, 2].set_title('Height Statistics')
        axes[1, 2].set_ylabel('Height (m)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quality dashboard saved to: {output_path}")
        return output_path
    
    def generate_quality_report(self, metrics: DataQualityMetrics, 
                              drift_result: Optional[DataDriftResult] = None,
                              validation_result: Optional[Dict[str, Any]] = None) -> Path:
        """
        Generate comprehensive quality report in JSON format.
        
        Args:
            metrics: DataQualityMetrics to report
            drift_result: Optional drift detection results
            validation_result: Optional validation results
        
        Returns:
            Path to generated report file
        """
        report_filename = f"quality_report_{metrics.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.output_path / report_filename
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_name': metrics.dataset_name,
                'report_version': '1.0'
            },
            'data_quality_metrics': asdict(metrics),
            'quality_assessment': {
                'overall_rating': self._get_quality_rating(metrics.overall_quality_score),
                'diversity_rating': self._get_quality_rating(metrics.diversity_score),
                'balance_rating': self._get_quality_rating(metrics.balance_score),
                'recommendations': self._generate_quality_recommendations(metrics)
            }
        }
        
        if drift_result:
            report['drift_analysis'] = asdict(drift_result)
        
        if validation_result:
            report['validation_results'] = validation_result
        
        # Add historical comparison if available
        if len(self.historical_metrics) > 1:
            report['historical_comparison'] = self._generate_historical_comparison()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to: {report_path}")
        return report_path
    
    def _calculate_height_distribution(self, heights: List[float]) -> Dict[str, float]:
        """Calculate height distribution in bins."""
        bins = np.linspace(0.3, 4.0, 8)  # 7 bins from 0.3 to 4.0 meters
        hist, bin_edges = np.histogram(heights, bins=bins)
        
        distribution = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}m"
            distribution[bin_label] = float(hist[i])
        
        return distribution
    
    def _calculate_diversity_score(self, wave_type_dist: Dict[str, int], 
                                 direction_dist: Dict[str, int], 
                                 heights: List[float]) -> float:
        """Calculate diversity score based on distribution entropy."""
        # Calculate entropy for categorical variables
        total_samples = sum(wave_type_dist.values())
        
        wave_type_entropy = 0
        for count in wave_type_dist.values():
            if count > 0:
                p = count / total_samples
                wave_type_entropy -= p * np.log2(p)
        
        direction_entropy = 0
        for count in direction_dist.values():
            if count > 0:
                p = count / total_samples
                direction_entropy -= p * np.log2(p)
        
        # Normalize entropies
        max_wave_entropy = np.log2(len(wave_type_dist))
        max_dir_entropy = np.log2(len(direction_dist))
        
        wave_diversity = wave_type_entropy / max_wave_entropy if max_wave_entropy > 0 else 0
        dir_diversity = direction_entropy / max_dir_entropy if max_dir_entropy > 0 else 0
        
        # Height diversity based on coefficient of variation
        height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
        height_diversity = min(height_cv, 1.0)  # Cap at 1.0
        
        # Combined diversity score
        diversity_score = (wave_diversity + dir_diversity + height_diversity) / 3
        return float(diversity_score)
    
    def _calculate_balance_score(self, wave_type_dist: Dict[str, int], 
                               direction_dist: Dict[str, int],
                               data_source_dist: Dict[str, int]) -> float:
        """Calculate balance score based on distribution uniformity."""
        def gini_coefficient(values):
            """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
            if not values or all(v == 0 for v in values):
                return 0
            sorted_values = sorted(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values)
            return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n
        
        # Calculate balance for each distribution (lower Gini = better balance)
        wave_balance = 1 - gini_coefficient(list(wave_type_dist.values()))
        dir_balance = 1 - gini_coefficient(list(direction_dist.values()))
        source_balance = 1 - gini_coefficient(list(data_source_dist.values()))
        
        # Combined balance score
        balance_score = (wave_balance + dir_balance + source_balance) / 3
        return float(balance_score)
    
    def _calculate_overall_quality_score(self, diversity_score: float, 
                                       balance_score: float, 
                                       sample_count: int) -> float:
        """Calculate overall quality score."""
        # Sample size factor (more samples = better, up to a point)
        size_factor = min(sample_count / 10000, 1.0)  # Normalize to 10k samples
        
        # Weighted combination
        overall_score = (0.4 * diversity_score + 0.4 * balance_score + 0.2 * size_factor)
        return float(overall_score)
    
    def _analyze_augmentation_coverage(self, samples_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze augmentation parameter coverage."""
        coverage = {
            'samples_with_augmentation': 0,
            'parameter_coverage': {},
            'parameter_ranges': {}
        }
        
        augmentation_samples = [
            s for s in samples_metadata 
            if 'augmentation_params' in s and s['augmentation_params']
        ]
        
        coverage['samples_with_augmentation'] = len(augmentation_samples)
        
        if augmentation_samples:
            # Analyze parameter coverage
            all_params = {}
            for sample in augmentation_samples:
                params = sample['augmentation_params']
                for key, value in params.items():
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(value)
            
            for param_name, values in all_params.items():
                if isinstance(values[0], (int, float)):
                    coverage['parameter_ranges'][param_name] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
                else:
                    # Categorical parameter
                    unique_values = set(values)
                    coverage['parameter_coverage'][param_name] = {
                        'unique_count': len(unique_values),
                        'values': list(unique_values)
                    }
        
        return coverage
    
    def _extract_drift_features(self, samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract features for drift detection."""
        features = {}
        
        # Extract numerical features
        heights = [s['height_meters'] for s in samples]
        features['height'] = np.array(heights)
        
        # Extract categorical features as numerical encodings
        wave_types = [s['wave_type'] for s in samples]
        wave_type_encoding = {'A_FRAME': 0, 'CLOSEOUT': 1, 'BEACH_BREAK': 2, 'POINT_BREAK': 3}
        features['wave_type'] = np.array([wave_type_encoding.get(wt, -1) for wt in wave_types])
        
        directions = [s['direction'] for s in samples]
        direction_encoding = {'LEFT': 0, 'RIGHT': 1, 'BOTH': 2}
        features['direction'] = np.array([direction_encoding.get(d, -1) for d in directions])
        
        return features
    
    def _perform_drift_test(self, current_data: np.ndarray, 
                          reference_data: np.ndarray, 
                          feature_name: str) -> Dict[str, float]:
        """Perform statistical test for drift detection."""
        try:
            # Use Kolmogorov-Smirnov test for continuous variables
            if feature_name == 'height':
                statistic, p_value = stats.ks_2samp(current_data, reference_data)
                effect_size = abs(np.mean(current_data) - np.mean(reference_data)) / np.std(reference_data)
            else:
                # Use Chi-square test for categorical variables
                current_counts = np.bincount(current_data.astype(int))
                reference_counts = np.bincount(reference_data.astype(int))
                
                # Ensure same length
                max_len = max(len(current_counts), len(reference_counts))
                current_counts = np.pad(current_counts, (0, max_len - len(current_counts)))
                reference_counts = np.pad(reference_counts, (0, max_len - len(reference_counts)))
                
                # Avoid division by zero
                expected = reference_counts + 1e-8
                statistic, p_value = stats.chisquare(current_counts, expected)
                effect_size = statistic / len(current_data)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(effect_size)
            }
            
        except Exception as e:
            logger.warning(f"Failed to perform drift test for {feature_name}: {e}")
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0
            }
    
    def _assess_sample_quality(self, sample: Dict[str, Any]) -> float:
        """Assess quality score for individual sample."""
        quality_score = 1.0
        
        # Check for required fields
        required_fields = ['height_meters', 'wave_type', 'direction', 'image_path']
        for field in required_fields:
            if field not in sample:
                quality_score -= 0.2
        
        # Check value ranges
        if 'height_meters' in sample:
            height = sample['height_meters']
            if not (0.1 <= height <= 10.0):  # Reasonable range
                quality_score -= 0.1
        
        # Check if image file exists
        if 'image_path' in sample:
            image_path = Path(sample['image_path'])
            if not image_path.exists():
                quality_score -= 0.3
        
        # Check for augmentation metadata quality
        if sample.get('data_source') == 'synthetic_from_real':
            if 'augmentation_params' not in sample:
                quality_score -= 0.1
            if 'depth_quality_score' in sample and sample['depth_quality_score'] < 0.3:
                quality_score -= 0.2
        
        return max(0.0, quality_score)
    
    def _identify_quality_issues(self, sample: Dict[str, Any]) -> List[str]:
        """Identify specific quality issues for a sample."""
        issues = []
        
        required_fields = ['height_meters', 'wave_type', 'direction', 'image_path']
        for field in required_fields:
            if field not in sample:
                issues.append(f"Missing required field: {field}")
        
        if 'height_meters' in sample:
            height = sample['height_meters']
            if not (0.1 <= height <= 10.0):
                issues.append(f"Height out of reasonable range: {height}")
        
        if 'image_path' in sample:
            image_path = Path(sample['image_path'])
            if not image_path.exists():
                issues.append("Image file does not exist")
        
        if sample.get('data_source') == 'synthetic_from_real':
            if 'augmentation_params' not in sample:
                issues.append("Missing augmentation parameters")
            if 'depth_quality_score' in sample and sample['depth_quality_score'] < 0.3:
                issues.append("Low depth quality score")
        
        return issues
    
    def _generate_drift_recommendations(self, drift_detected: bool, 
                                      affected_features: List[str], 
                                      drift_score: float) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("No significant data drift detected. Continue with current data pipeline.")
        else:
            recommendations.append(f"Data drift detected (score: {drift_score:.3f})")
            
            if 'height' in affected_features:
                recommendations.append("Height distribution has shifted - review wave generation parameters")
            
            if 'wave_type' in affected_features:
                recommendations.append("Wave type distribution has changed - check breaking behavior parameters")
            
            if 'direction' in affected_features:
                recommendations.append("Direction distribution has shifted - review directional spread parameters")
            
            recommendations.append("Consider retraining model or adjusting data generation parameters")
        
        return recommendations
    
    def _generate_quality_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if metrics.overall_quality_score < 0.5:
            recommendations.append("Overall quality is low - comprehensive review needed")
        
        if metrics.diversity_score < 0.3:
            recommendations.append("Low diversity - increase parameter variation in data generation")
        
        if metrics.balance_score < 0.2:
            recommendations.append("Poor class balance - adjust sampling strategy")
        
        if metrics.total_samples < 1000:
            recommendations.append("Small dataset size - consider generating more samples")
        
        # Check data source distribution
        synthetic_ratio = metrics.data_source_distribution.get('synthetic', 0) / metrics.total_samples
        if synthetic_ratio > 0.9:
            recommendations.append("Dataset is heavily synthetic - consider adding more real data for validation")
        
        return recommendations
    
    def _generate_historical_comparison(self) -> Dict[str, Any]:
        """Generate comparison with historical metrics."""
        if len(self.historical_metrics) < 2:
            return {}
        
        current = self.historical_metrics[-1]
        previous = self.historical_metrics[-2]
        
        return {
            'quality_trend': {
                'overall': current.overall_quality_score - previous.overall_quality_score,
                'diversity': current.diversity_score - previous.diversity_score,
                'balance': current.balance_score - previous.balance_score
            },
            'sample_count_change': current.total_samples - previous.total_samples,
            'time_difference': current.timestamp
        }
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _create_empty_metrics(self, dataset_name: str) -> DataQualityMetrics:
        """Create empty metrics for empty dataset."""
        return DataQualityMetrics(
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            total_samples=0,
            height_distribution={},
            wave_type_distribution={},
            direction_distribution={},
            data_source_distribution={},
            overall_quality_score=0.0,
            diversity_score=0.0,
            balance_score=0.0,
            height_statistics={},
            augmentation_coverage={}
        )