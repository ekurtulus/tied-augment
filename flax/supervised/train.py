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

"""ImageNet example.
This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

from audioop import cross
import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import input_pipeline
import models
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple


def create_model(*, model_cls, half_precision, num_classes, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (2, image_size, image_size, 3)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables['params'], variables['batch_stats']
 
def l2_loss(first_features, second_features):
    return jnp.mean((first_features - second_features) ** 2)

def l1_loss(first_features, second_features):
    return jnp.mean(jnp.abs(first_features - second_features))

def l2_normalized(first_features, second_features):
    return l2_loss( first_features / jnp.linalg.norm(first_features), 
                    second_features / jnp.linalg.norm(second_features), 
                  )

def cosine_similarity(first_features, second_features):
    return jnp.mean(optax.cosine_similarity(first_features, second_features))

def cross_entropy_double(first_probs, second_probs, first_labels, 
                         second_labels, both_branches_supervised=False):
    ce = cross_entropy_loss(first_probs, first_labels)
    if both_branches_supervised:
        ce = (ce + cross_entropy_loss(second_probs, second_labels) ) / 2
    return ce    

def cross_entropy_loss(probs, labels, dtype=jnp.float32):
    """Compute cross entropy for logits and labels w/ label smoothing
    Args:
        logits: [batch, length, num_classes] float array.
        labels: categorical labels [batch, length] int array.
        label_smoothing: label smoothing constant, used to determine the on and off values.
        dtype: dtype to perform loss calcs in, including log_softmax
    """
    return -jnp.mean(jnp.sum(probs * labels, axis=-1))


@partial(jax.jit, static_argnums=(7,8,9,10,11,12))
def criterion(first_logits, second_logits, first_features, second_features, first_labels,
              second_labels, similarity_weight=1.0, both_branches_supervised=False, 
              cross_entropy=True, similarity=True, loss_type="l2", logit_consistency_weight=0.0,
              logit_consistency_fn="js_div"):
    
    if loss_type == "cosine":
        similarity_fn = cosine_similarity
    elif loss_type == "l1":
        similarity_fn = l1_loss
    elif loss_type == "l2":
        similarity_fn = l2_loss
    elif loss_type == "l2_normalized":
        similarity_fn = l2_normalized
    
    first_probs = jax.nn.log_softmax(first_logits.astype(jnp.float32))
    second_probs = jax.nn.log_softmax(second_logits.astype(jnp.float32))
    
    if not similarity and cross_entropy:
        return cross_entropy_double(first_probs, second_probs, first_labels, second_labels, 
                                    both_branches_supervised=both_branches_supervised)
    
    elif similarity and not cross_entropy:
        return similarity_fn(first_features, second_features)
    
    else:
        return cross_entropy_double(first_probs, second_probs, first_labels, second_labels,
                                    both_branches_supervised=both_branches_supervised) + similarity_weight * similarity_fn(first_features, second_features)
    

def acc_topk(logits, labels, topk=(1,5)):
    top = lax.top_k(logits, max(topk))[1].transpose()
    correct = top == labels.reshape(1, -1)
    return [correct[:k].reshape(-1).sum(axis=0) * 100 / labels.shape[0] for k in topk]

def compute_eval_metrics(logits, features, labels):
  loss = cross_entropy_loss(jax.nn.log_softmax(logits.astype(jnp.float32)), 
                            jax.nn.one_hot(labels, logits.shape[-1], dtype=jnp.float32))  
  
  top_1, top_5 = acc_topk(logits, labels)
  metrics = {
      'loss': loss,
      'top-1' : top_1,
      'top-5' : top_5,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics

def compute_train_metrics(first_logits, first_features, second_logits, second_features, labels):

  loss = cross_entropy_loss(jax.nn.log_softmax(first_logits.astype(jnp.float32)), 
                            jax.nn.one_hot(labels, first_logits.shape[-1], dtype=jnp.float32))  
  
  top_1, top_5 = acc_topk(first_logits, labels)
    
  metrics = {
      'loss': loss,
      'top-1' : top_1,
      'top-5' : top_5,
      'l2' : (first_features - second_features) ** 2,
      'l1' : jnp.abs(first_features - second_features),
      'cosine' : optax.cosine_similarity(first_features, second_features),
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
    schedule,
    config,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_epochs = config.warmup_epochs
  num_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  
  if schedule == "constant":
    def schedule(count):
      return base_learning_rate
    scheduler = schedule 

  elif schedule == "cosine":
    scheduler = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=num_epochs * steps_per_epoch)
  
  elif schedule ==  "linear_up":
    scheduler = optax.linear_schedule(
            init_value=0,
            end_value=base_learning_rate,
            transition_steps=num_epochs * steps_per_epoch,
        )

  elif schedule ==  "linear_down":
    scheduler = optax.linear_schedule(
            init_value=base_learning_rate,
            end_value=0,
            transition_steps=num_epochs * steps_per_epoch,
        )

  elif schedule == "negate_odd":
    def schedule(count):
      if count % 2 == 0:
        return base_learning_rate
      else:
        return -base_learning_rate
    scheduler = schedule 
  elif schedule == "negate_even":
    def schedule(count):
      if count % 2 == 0:
        return -base_learning_rate
      else:
        return base_learning_rate
    scheduler = schedule 
  else:
    raise ValueError("unknown scheduler : ", schedule)

  if config.warmup_epochs > 0:
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, scheduler],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn
  return scheduler


def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient

def mixup(key, alpha, images, labels):
  lam = jax.random.beta(key, alpha, alpha)
  mixed_images = lam * images + (1. - lam) * images[::-1]
  mixed_labels = lam * labels + (1. - lam) * labels[::-1]

  return mixed_images, mixed_labels, lam

def train_step(state, batch, key, learning_rate_fn, config, tw=-1):
  first_images, second_images = batch["image1"], batch["image2"]
  first_labels, second_labels = (jax.nn.one_hot(batch["label"], config.num_classes, dtype=jnp.float32), 
                                  jax.nn.one_hot(batch["label"], config.num_classes, dtype=jnp.float32))
      
  if config.mixup_alpha > 0:
    first_images, first_labels, lam = mixup(key, config.mixup_alpha, first_images, first_labels)  

  """Perform a single training step."""
  def loss_fn(params, cross_entropy=True, similarity=True, loss_type="l2"):
    """loss function used for training."""      
    if "SBN" in config.model or config.single_forward:
      images = jnp.concatenate((first_images, second_images))
      (features, logits), new_model_state = state.apply_fn(
          {'params': params, 'batch_stats': state.batch_stats},
          images,
          mutable=['batch_stats'])
      first_features, second_features = jnp.split(features , 2)
      first_logits, second_logits = jnp.split(logits , 2)
    else:    
      (first_features, first_logits), new_model_state = state.apply_fn(
          {'params': params, 'batch_stats': state.batch_stats},
          first_images,
          mutable=['batch_stats'])
      
      (second_features, second_logits), new_model_state = state.apply_fn(
          {'params': params, 
          'batch_stats': new_model_state["batch_stats"] if not config.no_second_step_bn_update else state.batch_stats},
          second_images,
          mutable=['batch_stats'])
    
    if config.mixup_alpha > 0:
      second_features = lam * second_features + (1 - lam) * second_features[::-1]

    loss = criterion(first_logits, second_logits, first_features, second_features, first_labels,
                     second_labels, both_branches_supervised=config.both_branches_supervised, 
                     similarity_weight=tw, cross_entropy=cross_entropy, similarity=similarity, 
                     loss_type=loss_type)

    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1)
    weight_penalty = config.weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, first_logits, first_features, second_logits, second_features)
  
  
  def get_sam_gradient(model, rho, first_step, second_step, loss_type):
    """Returns the gradient of the SAM loss loss, updated state and logits.
    See https://arxiv.org/abs/2010.01412 for more details.
    Args:
      model: The model that we are training.
      rho: Size of the perturbation.
    """
    # compute gradient on the whole batch
    
    first_args = {"cross_entropy" : "ce" in first_step, "similarity" : "similarity" in first_step, "loss_type" : loss_type}
    second_args = {"cross_entropy" : "ce" in second_step, "similarity" : "similarity" in second_step, "loss_type" : loss_type}
    
    (_, (inner_state, _, _, _, _)), grad = jax.value_and_grad(
        lambda m: loss_fn(m, **first_args), has_aux=True)(model)
    
    grad = dual_vector(grad)
    noised_model = jax.tree_util.tree_map(lambda a, b: a + rho * b,
                                     model, grad)
    
    (_, (_, first_logits, first_features, second_logits, second_features)), grad = jax.value_and_grad(
        lambda m: loss_fn(m, **second_args), has_aux=True)(noised_model)
    
    return (inner_state, first_logits, first_features, second_logits, second_features), grad
    
  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    if config.rho > 0:
        aux, grads = get_sam_gradient(state.params, config.rho, config.sam_first_step, 
                                     config.sam_second_step, config.loss_type)
        grads = lax.pmean(grads, axis_name='batch')
    else:    
        grad_fn = jax.value_and_grad(lambda m : loss_fn(m, loss_type=config.loss_type), has_aux=True)
        aux, grads = grad_fn(state.params)
        grads = lax.pmean(grads, axis_name='batch')
        aux = aux[1]
        
  new_model_state, first_logits, first_features, second_logits, second_features = aux
  metrics = compute_train_metrics(first_logits, first_features, second_logits, second_features, batch['label'])
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params),
        dynamic_scale=dynamic_scale)
    metrics['scale'] = dynamic_scale.scale
  
  print("train step compiled ! ")
  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  features, logits = state.apply_fn(
      variables, batch['image'], train=False, mutable=False)
  print("eval step compiled ! ")
  return compute_eval_metrics(logits, features, batch['label'])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache, config):
  ds = input_pipeline.create_split(
              dataset_builder, batch_size, train, image_size=image_size, dtype=dtype,
              cache=cache, config=config)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(rng, config, model, image_size, learning_rate_fn):

  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, batch_stats = initialized(rng, image_size, model)
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
  )
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale)
  return state


def train_and_evaluate(config) -> TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=config.workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(0)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    
    
  DATASETS = {
    "imagenet2012" : {"num_classes" : 1000, "default_image_size" : 224, 
                      "mean" : IMAGENET_MEAN, 
                      "std" : IMAGENET_STD},
    "cifar10" : {"num_classes" : 10, "default_image_size" : 32, 
                 "mean" : [0.4914 * 255, 0.4822 * 255, 0.4465 * 255], 
                 "std" : [0.2475 * 255, 0.2435 * 255, 0.2615 * 255]},
      
    "cifar100" : {"num_classes" : 100, "default_image_size" : 32, 
                  "mean" : [0.5070 * 255, 0.4865 * 255, 0.4409 * 255], 
                  "std" : [0.2673 * 255, 0.2564 * 255, 0.2761 * 255]},
    "svhn_cropped" : {"num_classes" : 10, "default_image_size" : 32, 
                  "mean" : [0.4309 * 255, 0.4302 * 255, 0.4463 * 255], 
                  "std" : [0.1975 * 255, 0.2002 * 255, 0.1981 * 255]},    
  }
  if config.dataset in DATASETS:
    dataset_config = DATASETS[config.dataset]
  else:
    warnings.warn("Given dataset is not recognized so setting the configuration to ImageNet")
    dataset_config = DATASETS["imagenet2012"]

  config.mean = dataset_config["mean"] if config.mean == [-1, -1, -1] else config.mean
  config.std = dataset_config["std"]  if config.std == [-1, -1, -1] else config.std
  config.num_classes = dataset_config["num_classes"] if config.num_classes == -1 else config.num_classes
  config.image_size = dataset_config["default_image_size"] if config.image_size == -1 else config.image_size
  
  dataset_builder = tfds.builder(config.dataset)
  dataset_builder.download_and_prepare()

  train_iter = create_input_iter(
      dataset_builder, local_batch_size, config.image_size, input_dtype, True,
          config.cache, config)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, config.image_size, input_dtype, False,
          config.cache, config)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation' if config.dataset not in ["cifar10", "cifar100", "svhn_cropped"] else 'test'].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10
  
  if config.dataset not in ["cifar10", "cifar100", "svhn_cropped"]:
    base_learning_rate = config.learning_rate * (config.batch_size / 256.)
  else:
    base_learning_rate = config.learning_rate
    
  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision,
      num_classes=config.num_classes)

  learning_rate_fn = create_learning_rate_fn(config.lr_schedule,
      config, base_learning_rate, steps_per_epoch)
  
  tw_scheduler = create_learning_rate_fn(config.tw_schedule, 
                                         config,
                                         config.similarity_weight, steps_per_epoch)

  state = create_train_state(rng, config, model, config.image_size, learning_rate_fn)
  state = restore_checkpoint(state, config.workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn, config=config),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=config.workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    rng, key = random.split(rng)
    key = jax_utils.replicate(key)
    tw = tw_scheduler(step)
    tw = jax_utils.replicate(tw)

    state, metrics = p_train_step(state, batch, key=key, tw=tw)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.log_every_steps:
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, top-1 accuracy: %.2f, top-5 accuracy %.2f',
                   epoch, summary['loss'], summary['top-1'], summary['top-5'], )
      writer.write_scalars(
          step + 1, {f'eval_{key}': val for key, val in summary.items()})
      writer.flush()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, config.workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state
    