# Copyright 2022 The Flax Authors.
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

"""Main file for running the ImageNet example.
This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""
import os

#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["XLA_FLAGS"]="--xla_gpu_strict_conv_algorithm_picker=false"

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

import train
import warnings



FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('model', 'ResNet50', 'model name')
flags.DEFINE_string('dataset', 'imagenet2012', 'dataset name')

flags.DEFINE_integer('image_size', -1, 'image_size')
flags.DEFINE_multi_float('mean', [-1, -1, -1], 'mean of the dataset (optional)')
flags.DEFINE_multi_float('std', [-1, -1, -1], 'mean of the dataset (optional)')
flags.DEFINE_integer('num_classes', -1, 'number of classes in dataset (optional')

flags.DEFINE_boolean('same_crop', False, 'whether to use the same crop for two branches')

flags.DEFINE_string('lr_schedule', 'cosine', 'learning rate schedule')
flags.DEFINE_string('tw_schedule', 'constant', 'learning rate schedule')

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_float('warmup_epochs', 0.0, 'warmup epochs')
flags.DEFINE_float('momentum', 0.9, 'SGD momentum')
flags.DEFINE_integer('batch_size', 128, 'local batch size')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('label_smoothing', 0, 'label smoothing factor')
flags.DEFINE_float('ema_decay', 0, 'ema decay factor (ema is not applied if ema_decay is 0)')

flags.DEFINE_float('num_epochs', 180.0, 'number of epochs')
flags.DEFINE_integer('log_every_steps', 100, 'log every steps')
flags.DEFINE_boolean('cache', False, 'cache the dataset')
flags.DEFINE_boolean('half_precision', False, 'use half precision')

flags.DEFINE_integer('num_train_steps', -1, 'num train steps')
flags.DEFINE_integer('steps_per_eval', -1, 'steps per eval')

flags.DEFINE_boolean('both_branches_supervised', True, 'whether to apply CE to both left and right branches')
flags.DEFINE_float('similarity_weight', 1.0, 'weight of the cosine term')
flags.DEFINE_string('loss_type', 'l2', 'similarity loss type')
    
flags.DEFINE_boolean('single_forward', False, 'whether single forward pass is done instead of 2 forward passes')
flags.DEFINE_boolean('no_second_step_bn_update', False, 'whether single forward pass is done instead of 2 forward passes')
    
flags.DEFINE_string('first_transform', 'hflip-randaug_n2_m14_p1', 'augmentation for the left branch')
flags.DEFINE_string('second_transform', 'hflip-randaug_n2_m14_p1', 'augmentation for the right branch') 

flags.DEFINE_float('rho', 0, 'SAM rho (if 0, SAM is not applied)')
flags.DEFINE_string('sam_first_step', 'ce-similarity', 'first step of sam')
flags.DEFINE_string('sam_second_step', 'ce-similarity', 'second step of sam')

flags.DEFINE_float('mixup_alpha', 0.0, 'mixup alpha (if 0, no mixup is applied')
flags.DEFINE_string('finetuning_checkpoint', None, 'finetuning checkpoint to load from if the task is finetuning')
flags.DEFINE_boolean('linear_eval', False, 'if finetuning, use linear eval if true else full fine-tuning')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train.train_and_evaluate(FLAGS)


if __name__ == '__main__':
  app.run(main)
  
  