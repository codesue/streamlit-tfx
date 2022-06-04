import base64
from typing import Any, Callable, Dict, List, Optional, Text, Union
from textwrap import dedent

from ipywidgets.embed import embed_snippet
import pandas as pd
import streamlit as st
from tensorflow_data_validation.types import FeaturePath
from tensorflow_data_validation.utils.display_util import _get_combined_statistics
from tensorflow_data_validation.utils.display_util import (
    get_anomalies_dataframe, get_schema_dataframe)
from tensorflow_metadata.proto.v0.anomalies_pb2 import Anomalies
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList
from tensorflow_model_analysis.proto.config_pb2 import SlicingSpec
from tensorflow_model_analysis.slicer.slicer_lib import SingleSliceSpec
from tensorflow_model_analysis.view import (
    render_plot,
    render_slicing_attributions,
    render_slicing_metrics,
    render_time_series
)
from tensorflow_model_analysis.view.view_types import EvalResult, EvalResults

WRAPPER = dedent("""
    <div style="overflow-x: auto; border: 1px solid #e6e9ef;
      border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">
      {html}
    </div>
""")

FOOTER = dedent("""
    <span style="font-size: 0.75em">
      &hearts; Built with [`streamlit-tfx`](https://github.com/codesue/streamlit-tfx)
    </span>
""")

def wrap_html(html: str):
  """Wraps HTML components to make visualizations look more uniform."""
  return WRAPPER.format(html=html)

def get_widget_view_html(widget_view: Any) -> str:
  """Produces minimal HTML with an IPython widget view embedded."""
  snippet = embed_snippet(views=[widget_view])
  return wrap_html(snippet)

def display_widget_view(
    widget_view: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays a widget view.
  Args:
    widget_view: The widget view to display.
    width: Width of the frame in CSS pixels. Defaults to the app's default
      element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  """
  st.components.v1.html(
      html=get_widget_view_html(widget_view),
      width=width,
      height=height,
      scrolling=True
  )

def get_statistics_html(
    lhs_statistics: DatasetFeatureStatisticsList,
    rhs_statistics: Optional[DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[FeaturePath]] = None,
    denylist_features: Optional[List[FeaturePath]] = None
) -> Text:
  """Produces HTML for displaying input statistics using Facets.
  Args:
    lhs_statistics: A `DatasetFeatureStatisticsList` protocol buffer.
    rhs_statistics: A `DatasetFeatureStatisticsList` protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.
  Returns:
    HTML to be embedded for visualization.
  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protocol buffer has more than one dataset.
  """
  template = wrap_html("""
      <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
      <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
      <facets-overview proto-input="{protostr}"></facets-overview>
  """)
  combined_statistics = _get_combined_statistics(
      lhs_statistics, rhs_statistics, lhs_name, rhs_name, allowlist_features,
      denylist_features
  )
  protostr = base64.b64encode(
      combined_statistics.SerializeToString()
  ).decode('utf-8')

  return template.format(protostr=protostr)

def display_statistics(
    lhs_statistics: DatasetFeatureStatisticsList,
    rhs_statistics: Optional[DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[FeaturePath]] = None,
    denylist_features: Optional[List[FeaturePath]] = None,
    title: Optional[str] = 'Statistics',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays input statistics using Facets.
  Args:
    lhs_statistics: A `DatasetFeatureStatisticsList` protocol buffer.
    rhs_statistics: A `DatasetFeatureStatisticsList` protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.
    title: Title for the visualization. Defaults to "Statistics".
    width: Width of the frame in CSS pixels. Defaults to the app's default
      element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protocol buffer has more than one dataset.
  """
  assert (not allowlist_features or not denylist_features), (
      'Only specify one of allowlist_features and denylist_features.')

  if title:
    st.subheader(title)
  html = get_statistics_html(
      lhs_statistics,
      rhs_statistics,
      lhs_name, rhs_name,
      allowlist_features,
      denylist_features
  )
  st.components.v1.html(
      html=html,
      width=width,
      height=height,
      scrolling=True
  )

def display_schema(
    schema: Schema,
    title: Optional[str] = 'Schema',
    features_title: Optional[str] = 'Features',
    domains_title: Optional[str] = 'Domains',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays an input Schema.
  Args:
  schema: A `Schema` protocol buffer.
  title: Title for the visualization. Defaults to "Schema".
  features_title: Title for the features visualization. Defaults to "Features".
  domains_title: Title for the domain's visualization. Defaults to "Domains".
  width: Width of the frame in CSS pixels. Defaults to the app's default
    element width.
  height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  features_df, domains_df = get_schema_dataframe(schema)
  st.markdown(f'#### {features_title}')
  st.dataframe(features_df.style, width=width, height=height)
  if not domains_df.empty:
    pd.set_option('max_colwidth', None)
    st.markdown(f'#### {domains_title}')
    st.dataframe(domains_df.style, width=width, height=height)

def display_anomalies(
    anomalies: Anomalies,
    title: Optional[str] = 'Anomalies',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays input anomalies.
  Args:
  anomalies: An `Anomalies` protocol buffer.
  title: Title for the visualization. Defaults to "Anomalies".
  width: Width of the frame in CSS pixels. Defaults to the app's default
    element width.
  height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  anomalies_df = get_anomalies_dataframe(anomalies)
  if anomalies_df.empty:
    st.markdown('<h4 style="color:green;">No anomalies found.</h4>',
        unsafe_allow_html=True)
  else:
    st.markdown('<h4 style="color:red;">Found anomalies.</h4>',
        unsafe_allow_html=True)
    st.dataframe(anomalies_df.style, width=width, height=height)

def display_eval_result_plot(
    result: EvalResult,
    slicing_spec: Optional[Union[SingleSliceSpec, SlicingSpec]] = None,
    output_name: Optional[str] = None,
    class_id: Optional[int] = None,
    top_k: Optional[int] = None,
    k: Optional[int] = None,
    label: Optional[str] = None,
    title: Optional[str] = 'Evaluation Result Plot',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays an evaluation result plot.
  Args:
    result: A `tfma.EvalResult`.
    slicing_spec: The `tfma.SlicingSpec` to identify the slice. Show overall if
      unset.
    output_name: A string representing the output name.
    class_id: A number representing the class id if multi class.
    top_k: The k used to compute prediction in the top k position.
    k: The k used to compute the prediction at the kth position.
    label: A partial label used to match a set of plots in the results.
    title: Title for the visualization. Defaults to "Evaluation Result Plot".
    width: Width of the frame in CSS pixels. Defaults to the app's default
    element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  widget_view = render_plot(
      result=result,
      slicing_spec=slicing_spec,
      output_name=output_name,
      class_id=class_id,
      top_k=top_k,
      k=k,
      label=label,
  )
  display_widget_view(widget_view, height=height, width=width)

def display_eval_result_slicing_attributions(
    result: EvalResult,
    slicing_column: Optional[str] = None,
    slicing_spec: Optional[Union[SingleSliceSpec, SlicingSpec]] = None,
    metric_name: Optional[str] = None,
    weighted_example_column: Optional[str] = None,
    event_handlers: Optional[Callable[[Dict[str, Union[str, float]]], None]] = None,
    title: Optional[str] = 'Evaluation Result Slicing Attributions',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays evaluation result slicing attributions.
  Args:
    result: A `tfma.EvalResult`.
    slicing_column: The column to slice on.
    slicing_spec: The `tfma.SlicingSpec` to filter results. If neither column nor
      spec is set, show overall.
    metric_name: Name of attributions metric to show attributions for. Optional
      if only one metric used.
    weighted_example_column: Override for the weighted example column. This can
      be used when different weights are applied in different parts of the model
      (eg: multi-head).
    event_handlers: The event handlers.
    title: Title for the visualization. Defaults to "Evaluation Result Slicing
      Attributions".
    width: Width of the frame in CSS pixels. Defaults to the app's default
      element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  widget_view = render_slicing_attributions(
      result=result,
      slicing_column=slicing_column,
      slicing_spec=slicing_spec,
      metric_name=metric_name,
      weighted_example_column=weighted_example_column,
      event_handlers=event_handlers
  )
  display_widget_view(widget_view, height=height, width=width)

def display_eval_result_slicing_metrics(
    result: EvalResult,
    slicing_column: Optional[str] = None,
    slicing_spec: Optional[Union[SingleSliceSpec, SlicingSpec]] = None,
    weighted_example_column: Optional[str] = None,
    event_handlers: Optional[Callable[[Dict[str, Union[str, float]]], None]] = None,
    title: Optional[str] = 'Evaluation Result Slicing Metrics',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays evaluation result slicing metrics.
  Args:
    result: A `tfma.EvalResult`.
    slicing_column: The column to slice on.
    slicing_spec: The `tfma.SlicingSpec` to filter results. If neither column nor
      spec is set, show overall.
    weighted_example_column: Override for the weighted example column. This can
      be used when different weights are applied in different parts of the model
      (eg: multi-head).
    event_handlers: The event handlers.
    title: Title for the visualization. Defaults to "Evaluation Result Slicing
      Metrics".
    width: Width of the frame in CSS pixels. Defaults to the app's default
    element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  widget_view = render_slicing_metrics(
      result=result,
      slicing_column=slicing_column,
      slicing_spec=slicing_spec,
      weighted_example_column=weighted_example_column,
      event_handlers=event_handlers
  )
  display_widget_view(widget_view, height=height, width=width)

def display_eval_results_time_series(
    results: EvalResults,
    slicing_spec: Optional[Union[SingleSliceSpec, SlicingSpec]] = None,
    display_full_path: bool = False,
    title: Optional[str] = 'Evaluation Results Time Series',
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
  """Displays evaluation results time series.
  Args:
    results: A `tfma.EvalResults`.
    slicing_spec: A `tfma.SlicingSpec` determining the slice to show time series
      on. Show overall if not set.
    display_full_path: Whether to display the full path to the model / data in the
      visualization or just show a file name.
    width: Width of the frame in CSS pixels. Defaults to the app's default
      element width.
    height: Height of the frame in CSS pixels. Defaults to 150.
  """
  if title:
    st.subheader(title)
  widget_view = render_time_series(
      results=results,
      slicing_spec=slicing_spec,
      display_full_path=display_full_path
  )
  display_widget_view(widget_view, height=height, width=width)
