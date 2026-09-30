"""
Shot Peening Surface Deformation Detection module.
"""

from .model import (
    create_model,
    get_model_summary,
    set_encoder_trainable,
    get_parameter_groups
)

from .utils import (
    calculate_metrics,
    MetricTracker,
    EarlyStopping,
    create_overlay,
    visualize_prediction,
    generate_coverage_report,
    save_checkpoint,
    load_checkpoint,
    setup_logging
)

__all__ = [
    'create_model',
    'get_model_summary',
    'set_encoder_trainable',
    'get_parameter_groups',
    'calculate_metrics',
    'MetricTracker',
    'EarlyStopping',
    'create_overlay',
    'visualize_prediction',
    'generate_coverage_report',
    'save_checkpoint',
    'load_checkpoint',
    'setup_logging'
]
