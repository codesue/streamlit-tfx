import os

import streamlit as st
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tfx.utils import io_utils

import streamlit_tfx as st_tfx

def load_anomalies_proto(anomalies_path: str) -> anomalies_pb2.Anomalies:
  """Loads an anomalies proto from a file."""
  anomalies = anomalies_pb2.Anomalies()
  anomalies_bytes = io_utils.read_bytes_file(anomalies_path)
  anomalies.ParseFromString(anomalies_bytes)  # type: ignore
  return anomalies

st.set_page_config(
  page_title='streamlit-tfx',
  page_icon=':seedling:',
  layout='wide',
  initial_sidebar_state='auto',
  menu_items={
    'About': 'streamlit-tfx: TensorFlow Extended visualizers for Streamlit apps',
    'Get Help': None,
    'Report a bug': None,
  }
)

st.title('streamlit-tfx: TensorFlow Extended visualizers for Streamlit apps')
st.markdown('''
`streamlit-tfx` provides utilities for visualizing [TensorFlow Extended](https://www.tensorflow.org/tfx)
artifacts in [Streamlit](https://streamlit.io) apps.
''')
st.info('''🌱 Just sprouting! This project is in the very beginning stages of
development.''')

st.header('Installation')
st.code('pip install streamlit-tfx', language='shell')

st.header('Getting Started')
st.code('''
import streamlit_tfx as st_tfx

st_tfx.display(item)
st_tfx.display_statistics(statistics)
st_tfx.display_schema(schema)
st_tfx.display_anomalies(anomalies)
st_tfx.display_eval_result_plot(eval_result)
st_tfx.display_eval_result_slicing_attributions(eval_result)
st_tfx.display_eval_result_slicing_metrics(eval_result)
st_tfx.display_eval_results_time_series(eval_results)
''')

st.header('Using `streamlit-tfx` to display TFX pipeline artifacts')
st.markdown('''
Most artifacts used here were generated by running the
[TFX Keras Component tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras).
The anomalies artifact with anomalies was generated by running the
[TensorFlow Model Analysis tutorial](https://www.tensorflow.org/tfx/tutorials/model_analysis/tfma_basic).
''')
artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

# Display statistics
statistics_path = os.path.join(artifacts_dir, 'statistics/FeatureStats.pb')
statistics = tfdv.DatasetListView(tfdv.load_stats_binary(statistics_path)).proto()
st_tfx.display(statistics, height=600)

# Display schema
schema_path = os.path.join(artifacts_dir, 'schema/schema.pbtxt')
schema = tfdv.load_schema_text(schema_path)
st_tfx.display(schema)

# Display anomalies
no_anomalies_path = os.path.join(artifacts_dir, 'anomalies/no_anomalies/SchemaDiff.pb')
no_anomalies = load_anomalies_proto(no_anomalies_path)
st_tfx.display(no_anomalies, title='Artifact Without Anomalies')

has_anomalies_path = os.path.join(artifacts_dir, 'anomalies/has_anomalies/SchemaDiff.pb')
has_anomalies = load_anomalies_proto(has_anomalies_path)
st_tfx.display(has_anomalies, title='Artifact With Anomalies')

# Display evaluation results
evaluation_path = os.path.join(artifacts_dir, 'evaluation')
eval_result = tfma.load_eval_result(evaluation_path)
st_tfx.display(eval_result, height=700)

eval_results = tfma.make_eval_results(
    results=[eval_result], mode=tfma.DATA_CENTRIC_MODE)
st_tfx.display(eval_results, height=600)

# TODO: st_tfx.display_eval_result_plot(eval_result)  # pylint: disable=fixme

# TODO: st_tfx.display_eval_result_slicing_attributions(eval_result)  # pylint: disable=fixme
