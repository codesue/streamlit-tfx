from typing import Any, Callable

import streamlit as st
from tensorflow_metadata.proto.v0.anomalies_pb2 import Anomalies
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList
from tensorflow_model_analysis.view.view_types import EvalResult, EvalResults

from streamlit_tfx.visualization import (
    display_anomalies,
    display_schema,
    display_statistics,
    display_eval_result_plot,
    display_eval_result_slicing_attributions,
    display_eval_result_slicing_metrics,
    display_eval_results_time_series,
)

__version__ = '22.6.4-dev'

__all__ = [
    'display',
    'display_anomalies',
    'display_schema',
    'display_statistics',
    'display_eval_result_plot',
    'display_eval_result_slicing_attributions',
    'display_eval_result_slicing_metrics',
    'display_eval_results_time_series',
]

_VISUALIZATIONS = {
    Anomalies: display_anomalies,
    DatasetFeatureStatisticsList: display_statistics,
    EvalResult: display_eval_result_slicing_metrics,
    EvalResults: display_eval_results_time_series,
    Schema: display_schema,
}

def _get_visualizer(item: Any) -> Callable:
  """Returns a visualizer to display an item.

  Args:
    item: The item to display.
  """
  visualizer = _VISUALIZATIONS.get(type(item), st.write)
  return visualizer

def display(item, **kwargs) -> None:
  """Displays an item.

  Args:
    item: The item to display.
  """
  visualizer = _get_visualizer(item)
  visualizer(item, **kwargs)
