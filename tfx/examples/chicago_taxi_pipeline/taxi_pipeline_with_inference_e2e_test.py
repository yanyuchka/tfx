# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_with_inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_with_inference
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


class TaxiPipelineWithInferenceEndToEndTest(tf.test.TestCase,
                                            parameterized.TestCase):

  def setUp(self):
    super(TaxiPipelineWithInferenceEndToEndTest, self).setUp() # pylint: disable=super-with-arguments
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'inference_test'
    self._training_data_root = os.path.join(os.path.dirname(__file__),
                                            'data',
                                            'simple')
    self._inference_data_root = os.path.join(os.path.dirname(__file__),
                                             'data',
                                             'unlabelled')
    self._module_file = os.path.join(os.path.dirname(__file__), 'taxi_utils.py')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    outputs = fileio.listdir(component_path)

    self.assertIn('.system', outputs)
    outputs.remove('.system')
    system_paths = [
        os.path.join('.system', path)
        for path in fileio.listdir(os.path.join(component_path, '.system'))
    ]
    self.assertNotEmpty(system_paths)
    self.assertIn('.system/executor_execution', system_paths)
    outputs.extend(system_paths)
    self.assertNotEmpty(outputs)
    for output in outputs:
      execution = fileio.listdir(os.path.join(component_path, output))
      self.assertLen(execution, 1)

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('CsvExampleGen.inference_example_gen')
    self.assertExecutedOnce('CsvExampleGen.training_example_gen')
    self.assertExecutedOnce('Evaluator')
    self.assertExecutedOnce('ExampleValidator')
    self.assertExecutedOnce('BulkInferrer')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('Trainer')
    self.assertExecutedOnce('Transform')

  def testTaxiPipelineBeam(self):
    BeamDagRunner().run(
            taxi_pipeline_with_inference._create_pipeline( # pylint: disable=protected-access
            pipeline_name=self._pipeline_name,
            training_data_root=self._training_data_root,
            inference_data_root=self._inference_data_root,
            module_file=self._module_file,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            beam_pipeline_args=[]))

    self.assertTrue(fileio.exists(self._metadata_path))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)

    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(10, execution_count)

    self.assertPipelineExecution()


if __name__ == '__main__':
  tf.test.main()
