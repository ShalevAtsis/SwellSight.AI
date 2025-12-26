"""Comprehensive metrics calculator for wave analysis model evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score, 
    f1_score, confusion_matrix, classification_report
)
import torch

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    mae: float
    rmse: float
    mse: float
    r2_score: float
    mean_error: float
    std_error: float


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray
    class_report: Dict[str, Any]


@dataclass
class EvaluationResults:
    """Container for complete evaluation results."""
    height_metrics: RegressionMetrics
    wave_type_metrics: ClassificationMetrics
    direction_metrics: ClassificationMetrics
    num_samples: int
    dataset_type: str


class MetricsCalculator:
    """
    Comprehensive calculator for evaluation metrics.
    
    Computes MAE, RMSE for height regression and accuracy, F1-score, 
    confusion matrices for classification tasks. Supports separate 
    evaluation on synthetic vs real data with performance reports 
    and visualizations.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        # Class mappings for interpretability
        self.wave_type_classes = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
        self.direction_classes = ['LEFT', 'RIGHT', 'BOTH']
        
        # Index to class mappings
        self.idx_to_wave_type = {i: cls for i, cls in enumerate(self.wave_type_classes)}
        self.idx_to_direction = {i: cls for i, cls in enumerate(self.direction_classes)}
        
        logger.info("Initialized MetricsCalculator")
    
    def _extract_predictions_and_targets(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract and format predictions and targets for metric calculation.
        
        Args:
            predictions: List of model predictions (WavePrediction objects or dicts)
            targets: List of ground truth targets (dicts or tensors)
        
        Returns:
            Tuple of (formatted_predictions, formatted_targets)
        """
        pred_heights = []
        pred_wave_types = []
        pred_directions = []
        
        target_heights = []
        target_wave_types = []
        target_directions = []
        
        for pred, target in zip(predictions, targets):
            # Handle prediction format (WavePrediction object or dict)
            if hasattr(pred, 'height_meters'):
                # WavePrediction object
                pred_heights.append(pred.height_meters)
                pred_wave_types.append(pred.wave_type)
                pred_directions.append(pred.direction)
            elif isinstance(pred, dict):
                # Dictionary format
                pred_heights.append(pred['height_meters'])
                pred_wave_types.append(pred['wave_type'])
                pred_directions.append(pred['direction'])
            else:
                raise ValueError(f"Unsupported prediction format: {type(pred)}")
            
            # Handle target format (dict or tensor)
            if isinstance(target, dict):
                if 'height' in target:
                    # Tensor format from DataLoader
                    target_heights.append(float(target['height']))
                    target_wave_types.append(int(target['wave_type']))
                    target_directions.append(int(target['direction']))
                else:
                    # Dict with string keys
                    target_heights.append(target['height_meters'])
                    # Convert string labels to indices
                    target_wave_types.append(self.wave_type_classes.index(target['wave_type']))
                    target_directions.append(self.direction_classes.index(target['direction']))
            else:
                raise ValueError(f"Unsupported target format: {type(target)}")
        
        # Convert string predictions to indices if needed
        pred_wave_type_indices = []
        pred_direction_indices = []
        
        for wave_type in pred_wave_types:
            if isinstance(wave_type, str):
                pred_wave_type_indices.append(self.wave_type_classes.index(wave_type))
            else:
                pred_wave_type_indices.append(int(wave_type))
        
        for direction in pred_directions:
            if isinstance(direction, str):
                pred_direction_indices.append(self.direction_classes.index(direction))
            else:
                pred_direction_indices.append(int(direction))
        
        formatted_predictions = {
            'height': np.array(pred_heights),
            'wave_type': np.array(pred_wave_type_indices),
            'direction': np.array(pred_direction_indices)
        }
        
        formatted_targets = {
            'height': np.array(target_heights),
            'wave_type': np.array(target_wave_types),
            'direction': np.array(target_directions)
        }
        
        return formatted_predictions, formatted_targets
    
    def calculate_regression_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> RegressionMetrics:
        """
        Calculate regression metrics for height prediction.
        
        Args:
            predictions: Predicted height values
            targets: Ground truth height values
        
        Returns:
            RegressionMetrics object with computed metrics
        """
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Error statistics
        errors = predictions - targets
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return RegressionMetrics(
            mae=mae,
            rmse=rmse,
            mse=mse,
            r2_score=r2_score,
            mean_error=mean_error,
            std_error=std_error
        )
    
    def calculate_classification_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        class_names: List[str]
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.
        
        Args:
            predictions: Predicted class indices
            targets: Ground truth class indices
            class_names: List of class names for reporting
        
        Returns:
            ClassificationMetrics object with computed metrics
        """
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Precision and recall (weighted average)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report with labels parameter to handle missing classes
        unique_labels = sorted(list(set(targets) | set(predictions)))
        max_label = len(class_names) - 1
        all_labels = list(range(max_label + 1))
        
        class_report = classification_report(
            targets, predictions, 
            labels=all_labels,
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        return ClassificationMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            confusion_matrix=cm,
            class_report=class_report
        )
    
    def calculate_all_metrics(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]],
        dataset_type: str = "unknown"
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all evaluation metrics.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            dataset_type: Type of dataset ("synthetic" or "real")
        
        Returns:
            Dictionary of metrics organized by task
        """
        if not predictions or not targets:
            logger.warning("Empty predictions or targets provided")
            return {
                "height": {"mae": 0.0, "rmse": 0.0},
                "wave_type": {"accuracy": 0.0, "f1_score": 0.0},
                "direction": {"accuracy": 0.0, "f1_score": 0.0}
            }
        
        if len(predictions) != len(targets):
            raise ValueError(f"Mismatch in predictions ({len(predictions)}) and targets ({len(targets)}) length")
        
        # Extract and format data
        formatted_preds, formatted_targets = self._extract_predictions_and_targets(predictions, targets)
        
        # Calculate regression metrics for height
        height_metrics = self.calculate_regression_metrics(
            formatted_preds['height'], 
            formatted_targets['height']
        )
        
        # Calculate classification metrics for wave type
        wave_type_metrics = self.calculate_classification_metrics(
            formatted_preds['wave_type'],
            formatted_targets['wave_type'],
            self.wave_type_classes
        )
        
        # Calculate classification metrics for direction
        direction_metrics = self.calculate_classification_metrics(
            formatted_preds['direction'],
            formatted_targets['direction'],
            self.direction_classes
        )
        
        logger.info(f"Calculated metrics for {len(predictions)} samples ({dataset_type} dataset)")
        
        # Return in expected format for backward compatibility
        return {
            "height": {
                "mae": height_metrics.mae,
                "rmse": height_metrics.rmse,
                "mse": height_metrics.mse,
                "r2_score": height_metrics.r2_score,
                "mean_error": height_metrics.mean_error,
                "std_error": height_metrics.std_error
            },
            "wave_type": {
                "accuracy": wave_type_metrics.accuracy,
                "f1_score": wave_type_metrics.f1_score,
                "precision": wave_type_metrics.precision,
                "recall": wave_type_metrics.recall
            },
            "direction": {
                "accuracy": direction_metrics.accuracy,
                "f1_score": direction_metrics.f1_score,
                "precision": direction_metrics.precision,
                "recall": direction_metrics.recall
            }
        }
    
    def calculate_detailed_metrics(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]],
        dataset_type: str = "unknown"
    ) -> EvaluationResults:
        """
        Calculate detailed evaluation metrics with full results.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            dataset_type: Type of dataset ("synthetic" or "real")
        
        Returns:
            EvaluationResults object with detailed metrics
        """
        if not predictions or not targets:
            raise ValueError("Empty predictions or targets provided")
        
        if len(predictions) != len(targets):
            raise ValueError(f"Mismatch in predictions ({len(predictions)}) and targets ({len(targets)}) length")
        
        # Extract and format data
        formatted_preds, formatted_targets = self._extract_predictions_and_targets(predictions, targets)
        
        # Calculate all metrics
        height_metrics = self.calculate_regression_metrics(
            formatted_preds['height'], 
            formatted_targets['height']
        )
        
        wave_type_metrics = self.calculate_classification_metrics(
            formatted_preds['wave_type'],
            formatted_targets['wave_type'],
            self.wave_type_classes
        )
        
        direction_metrics = self.calculate_classification_metrics(
            formatted_preds['direction'],
            formatted_targets['direction'],
            self.direction_classes
        )
        
        return EvaluationResults(
            height_metrics=height_metrics,
            wave_type_metrics=wave_type_metrics,
            direction_metrics=direction_metrics,
            num_samples=len(predictions),
            dataset_type=dataset_type
        )
    
    def evaluate_by_height_range(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]],
        height_ranges: List[Tuple[float, float]] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate performance across different wave height ranges.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            height_ranges: List of (min, max) height ranges to evaluate
        
        Returns:
            Dictionary of metrics by height range
        """
        if height_ranges is None:
            height_ranges = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
        
        # Extract and format data
        formatted_preds, formatted_targets = self._extract_predictions_and_targets(predictions, targets)
        
        results = {}
        
        for min_height, max_height in height_ranges:
            range_name = f"{min_height:.1f}-{max_height:.1f}m"
            
            # Filter samples in this height range
            mask = (formatted_targets['height'] >= min_height) & (formatted_targets['height'] < max_height)
            
            if not np.any(mask):
                logger.warning(f"No samples found in height range {range_name}")
                continue
            
            # Filter predictions and targets
            range_preds = {
                'height': formatted_preds['height'][mask],
                'wave_type': formatted_preds['wave_type'][mask],
                'direction': formatted_preds['direction'][mask]
            }
            
            range_targets = {
                'height': formatted_targets['height'][mask],
                'wave_type': formatted_targets['wave_type'][mask],
                'direction': formatted_targets['direction'][mask]
            }
            
            # Calculate metrics for this range
            height_metrics = self.calculate_regression_metrics(
                range_preds['height'], 
                range_targets['height']
            )
            
            wave_type_metrics = self.calculate_classification_metrics(
                range_preds['wave_type'],
                range_targets['wave_type'],
                self.wave_type_classes
            )
            
            direction_metrics = self.calculate_classification_metrics(
                range_preds['direction'],
                range_targets['direction'],
                self.direction_classes
            )
            
            results[range_name] = {
                "height": {
                    "mae": height_metrics.mae,
                    "rmse": height_metrics.rmse,
                    "r2_score": height_metrics.r2_score
                },
                "wave_type": {
                    "accuracy": wave_type_metrics.accuracy,
                    "f1_score": wave_type_metrics.f1_score
                },
                "direction": {
                    "accuracy": direction_metrics.accuracy,
                    "f1_score": direction_metrics.f1_score
                },
                "num_samples": int(np.sum(mask))
            }
        
        return results
    
    def generate_visualizations(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]],
        output_dir: Path,
        dataset_type: str = "unknown"
    ) -> None:
        """
        Generate evaluation visualizations.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            output_dir: Directory to save visualizations
            dataset_type: Type of dataset for labeling
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and format data
        formatted_preds, formatted_targets = self._extract_predictions_and_targets(predictions, targets)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Height regression scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(formatted_targets['height'], formatted_preds['height'], alpha=0.6)
        plt.plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
        plt.xlabel('Ground Truth Height (m)')
        plt.ylabel('Predicted Height (m)')
        plt.title(f'Height Prediction Accuracy ({dataset_type} dataset)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'height_scatter_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Height error distribution
        errors = formatted_preds['height'] - formatted_targets['height']
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (m)')
        plt.ylabel('Frequency')
        plt.title(f'Height Prediction Error Distribution ({dataset_type} dataset)')
        plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'height_error_dist_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Wave type confusion matrix
        wave_type_metrics = self.calculate_classification_metrics(
            formatted_preds['wave_type'],
            formatted_targets['wave_type'],
            self.wave_type_classes
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            wave_type_metrics.confusion_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=self.wave_type_classes,
            yticklabels=self.wave_type_classes,
            cmap='Blues'
        )
        plt.xlabel('Predicted Wave Type')
        plt.ylabel('True Wave Type')
        plt.title(f'Wave Type Confusion Matrix ({dataset_type} dataset)')
        plt.tight_layout()
        plt.savefig(output_dir / f'wave_type_confusion_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Direction confusion matrix
        direction_metrics = self.calculate_classification_metrics(
            formatted_preds['direction'],
            formatted_targets['direction'],
            self.direction_classes
        )
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            direction_metrics.confusion_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=self.direction_classes,
            yticklabels=self.direction_classes,
            cmap='Greens'
        )
        plt.xlabel('Predicted Direction')
        plt.ylabel('True Direction')
        plt.title(f'Direction Confusion Matrix ({dataset_type} dataset)')
        plt.tight_layout()
        plt.savefig(output_dir / f'direction_confusion_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Performance by height range
        height_range_results = self.evaluate_by_height_range(predictions, targets)
        
        if height_range_results:
            ranges = list(height_range_results.keys())
            height_maes = [height_range_results[r]['height']['mae'] for r in ranges]
            wave_type_accs = [height_range_results[r]['wave_type']['accuracy'] for r in ranges]
            direction_accs = [height_range_results[r]['direction']['accuracy'] for r in ranges]
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Height MAE by range
            ax1.bar(ranges, height_maes)
            ax1.set_xlabel('Height Range')
            ax1.set_ylabel('MAE (m)')
            ax1.set_title('Height MAE by Range')
            ax1.tick_params(axis='x', rotation=45)
            
            # Wave type accuracy by range
            ax2.bar(ranges, wave_type_accs)
            ax2.set_xlabel('Height Range')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Wave Type Accuracy by Range')
            ax2.tick_params(axis='x', rotation=45)
            
            # Direction accuracy by range
            ax3.bar(ranges, direction_accs)
            ax3.set_xlabel('Height Range')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Direction Accuracy by Range')
            ax3.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'performance_by_height_{dataset_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Generated visualizations in {output_dir}")
    
    def generate_report(
        self, 
        predictions: List[Union[Dict[str, Any], Any]], 
        targets: List[Union[Dict[str, Any], torch.Tensor]], 
        output_path: Path,
        dataset_type: str = "unknown"
    ) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            output_path: Path to save HTML report
            dataset_type: Type of dataset for labeling
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate detailed metrics
        results = self.calculate_detailed_metrics(predictions, targets, dataset_type)
        
        # Generate visualizations
        viz_dir = output_dir / 'visualizations'
        self.generate_visualizations(predictions, targets, viz_dir, dataset_type)
        
        # Calculate performance by height range
        height_range_results = self.evaluate_by_height_range(predictions, targets)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SwellSight Wave Analysis Model - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 100%; height: auto; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .poor {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SwellSight Wave Analysis Model - Evaluation Report</h1>
                <p><strong>Dataset Type:</strong> {dataset_type.title()}</p>
                <p><strong>Number of Samples:</strong> {results.num_samples}</p>
                <p><strong>Generated:</strong> {Path().cwd()}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance Summary</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Task</th>
                        <th>Primary Metric</th>
                        <th>Value</th>
                        <th>Assessment</th>
                    </tr>
                    <tr>
                        <td>Height Regression</td>
                        <td>MAE (meters)</td>
                        <td>{results.height_metrics.mae:.3f}</td>
                        <td class="{'good' if results.height_metrics.mae < 0.3 else 'warning' if results.height_metrics.mae < 0.5 else 'poor'}">
                            {'Excellent' if results.height_metrics.mae < 0.3 else 'Good' if results.height_metrics.mae < 0.5 else 'Needs Improvement'}
                        </td>
                    </tr>
                    <tr>
                        <td>Wave Type Classification</td>
                        <td>Accuracy</td>
                        <td>{results.wave_type_metrics.accuracy:.3f}</td>
                        <td class="{'good' if results.wave_type_metrics.accuracy > 0.8 else 'warning' if results.wave_type_metrics.accuracy > 0.6 else 'poor'}">
                            {'Excellent' if results.wave_type_metrics.accuracy > 0.8 else 'Good' if results.wave_type_metrics.accuracy > 0.6 else 'Needs Improvement'}
                        </td>
                    </tr>
                    <tr>
                        <td>Direction Classification</td>
                        <td>Accuracy</td>
                        <td>{results.direction_metrics.accuracy:.3f}</td>
                        <td class="{'good' if results.direction_metrics.accuracy > 0.8 else 'warning' if results.direction_metrics.accuracy > 0.6 else 'poor'}">
                            {'Excellent' if results.direction_metrics.accuracy > 0.8 else 'Good' if results.direction_metrics.accuracy > 0.6 else 'Needs Improvement'}
                        </td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Detailed Height Regression Metrics</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean Absolute Error (MAE)</td><td>{results.height_metrics.mae:.4f} meters</td></tr>
                    <tr><td>Root Mean Square Error (RMSE)</td><td>{results.height_metrics.rmse:.4f} meters</td></tr>
                    <tr><td>Mean Square Error (MSE)</td><td>{results.height_metrics.mse:.4f}</td></tr>
                    <tr><td>R² Score</td><td>{results.height_metrics.r2_score:.4f}</td></tr>
                    <tr><td>Mean Error (Bias)</td><td>{results.height_metrics.mean_error:.4f} meters</td></tr>
                    <tr><td>Standard Deviation of Error</td><td>{results.height_metrics.std_error:.4f} meters</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Wave Type Classification Metrics</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Accuracy</td><td>{results.wave_type_metrics.accuracy:.4f}</td></tr>
                    <tr><td>F1-Score (Weighted)</td><td>{results.wave_type_metrics.f1_score:.4f}</td></tr>
                    <tr><td>Precision (Weighted)</td><td>{results.wave_type_metrics.precision:.4f}</td></tr>
                    <tr><td>Recall (Weighted)</td><td>{results.wave_type_metrics.recall:.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Direction Classification Metrics</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Accuracy</td><td>{results.direction_metrics.accuracy:.4f}</td></tr>
                    <tr><td>F1-Score (Weighted)</td><td>{results.direction_metrics.f1_score:.4f}</td></tr>
                    <tr><td>Precision (Weighted)</td><td>{results.direction_metrics.precision:.4f}</td></tr>
                    <tr><td>Recall (Weighted)</td><td>{results.direction_metrics.recall:.4f}</td></tr>
                </table>
            </div>
        """
        
        # Add height range analysis if available
        if height_range_results:
            html_content += """
            <div class="section">
                <h2>Performance by Height Range</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Height Range</th>
                        <th>Samples</th>
                        <th>Height MAE</th>
                        <th>Wave Type Acc</th>
                        <th>Direction Acc</th>
                    </tr>
            """
            
            for range_name, metrics in height_range_results.items():
                html_content += f"""
                    <tr>
                        <td>{range_name}</td>
                        <td>{metrics['num_samples']}</td>
                        <td>{metrics['height']['mae']:.3f}m</td>
                        <td>{metrics['wave_type']['accuracy']:.3f}</td>
                        <td>{metrics['direction']['accuracy']:.3f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add visualizations
        html_content += """
            <div class="section">
                <h2>Visualizations</h2>
                
                <div class="visualization">
                    <h3>Height Prediction Accuracy</h3>
                    <img src="visualizations/height_scatter_{}.png" alt="Height Scatter Plot">
                </div>
                
                <div class="visualization">
                    <h3>Height Prediction Error Distribution</h3>
                    <img src="visualizations/height_error_dist_{}.png" alt="Height Error Distribution">
                </div>
                
                <div class="visualization">
                    <h3>Wave Type Confusion Matrix</h3>
                    <img src="visualizations/wave_type_confusion_{}.png" alt="Wave Type Confusion Matrix">
                </div>
                
                <div class="visualization">
                    <h3>Direction Confusion Matrix</h3>
                    <img src="visualizations/direction_confusion_{}.png" alt="Direction Confusion Matrix">
                </div>
        """.format(dataset_type, dataset_type, dataset_type, dataset_type)
        
        if height_range_results:
            html_content += f"""
                <div class="visualization">
                    <h3>Performance by Height Range</h3>
                    <img src="visualizations/performance_by_height_{dataset_type}.png" alt="Performance by Height Range">
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated comprehensive evaluation report: {output_path}")
    
    def compare_datasets(
        self, 
        synthetic_predictions: List[Union[Dict[str, Any], Any]], 
        synthetic_targets: List[Union[Dict[str, Any], torch.Tensor]],
        real_predictions: List[Union[Dict[str, Any], Any]], 
        real_targets: List[Union[Dict[str, Any], torch.Tensor]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Compare performance between synthetic and real datasets.
        
        Args:
            synthetic_predictions: Predictions on synthetic data
            synthetic_targets: Targets for synthetic data
            real_predictions: Predictions on real data
            real_targets: Targets for real data
            output_dir: Directory to save comparison results
        
        Returns:
            Dictionary with comparison results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics for both datasets
        synthetic_metrics = self.calculate_all_metrics(
            synthetic_predictions, synthetic_targets, "synthetic"
        )
        real_metrics = self.calculate_all_metrics(
            real_predictions, real_targets, "real"
        )
        
        # Calculate performance gaps
        comparison = {
            "synthetic_metrics": synthetic_metrics,
            "real_metrics": real_metrics,
            "performance_gaps": {
                "height_mae_gap": real_metrics["height"]["mae"] - synthetic_metrics["height"]["mae"],
                "wave_type_accuracy_gap": synthetic_metrics["wave_type"]["accuracy"] - real_metrics["wave_type"]["accuracy"],
                "direction_accuracy_gap": synthetic_metrics["direction"]["accuracy"] - real_metrics["direction"]["accuracy"]
            },
            "domain_adaptation_needed": {
                "height": real_metrics["height"]["mae"] > synthetic_metrics["height"]["mae"] * 1.5,
                "wave_type": real_metrics["wave_type"]["accuracy"] < synthetic_metrics["wave_type"]["accuracy"] * 0.8,
                "direction": real_metrics["direction"]["accuracy"] < synthetic_metrics["direction"]["accuracy"] * 0.8
            }
        }
        
        # Save comparison results
        with open(output_dir / 'dataset_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Generated dataset comparison results in {output_dir}")
        
        return comparison