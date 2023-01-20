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

"""ImageNet input pipeline.
"""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import augmentations
from functools import partial
import logging
IMAGE_SIZE = 224
CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.io.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.io.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: _resize(image, image_size))

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.io.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image, mean, std):
  image = tf.cast(image, tf.float32)
  image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, config=None):
  """Preprocesses the given image for evaluation.
  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.
  Returns:
    A preprocessed image `Tensor`.
  """
  if config.dataset not in ["cifar10", "cifar100", "svhn_cropped"]:
    image = _decode_and_center_crop(image_bytes, image_size)
  else:
    image = tf.image.decode_jpeg(image_bytes, 3)
    
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image, config.mean, config.std)
  image = tf.image.convert_image_dtype(image, dtype=dtype)

  return image

def augmented_view(image_bytes, transform, image_size=IMAGE_SIZE, dtype=tf.float32,
             config=None, cutout_size=-1, no_decode=False):
  
  if not no_decode:
    image = tf.image.decode_jpeg(image_bytes, 3)
  else:
    image = image_bytes

  image = tf.reshape(image, [image_size, image_size, 3])
  image = transform(tf.cast(image, tf.uint8))
  image = normalize_image(image, config.mean, config.std)
  image = tf.image.convert_image_dtype(image, dtype)
  image = tf.cond(tf.math.not_equal(cutout_size, -1), lambda: augmentations.cutout(image, cutout_size), lambda : image)
  return image


def two_augmented_views(image_bytes, first_transform, second_transform, dtype=tf.float32, 
                        image_size=IMAGE_SIZE, config=None, cutout_sizes=(-1, -1)):
    
    if config.dataset not in ["cifar10", "cifar100", "svhn_cropped"]:
        first = _decode_and_random_crop(image_bytes, image_size)
        if not config.same_crop:
            second = _decode_and_random_crop(image_bytes, image_size) 
        else:
            second = first
    
    else:
        first = tf.image.decode_jpeg(image_bytes, 3)
        first = tf.image.random_crop(first, size=[28, 28, 3])
        first = tf.image.resize_with_crop_or_pad(first, 32, 32)
        if not config.same_crop:
            second = tf.image.decode_jpeg(image_bytes, 3)
            second = tf.image.random_crop(second, size=[28, 28, 3])
            second = tf.image.resize_with_crop_or_pad(second, 32, 32)
        else:
            second = first

    first = augmented_view(first, first_transform, image_size=image_size,
                                    dtype=dtype, config=config, cutout_size=cutout_sizes[0], no_decode=True)

    second = augmented_view(second, second_transform, image_size=image_size,
                                    dtype=dtype, config=config, cutout_size=cutout_sizes[1], no_decode=True)
    
    return first, second
    
def _solve_transform(transform_name):
    transforms = []
    transform_functions = transform_name.split("-")
    cutout_size = -1
    for transform in transform_functions:
        if transform == "hflip":
            transforms.append(tf.image.random_flip_left_right)
        elif transform.startswith("randaug"):
            _, num_layers, magnitude, prob = transform.split("_")
            transforms.append(partial(augmentations.distort_image_with_randaugment, 
                                      num_layers=int(num_layers[1:]), magnitude=int(magnitude[1:]),
                                      probability=float(prob[1:]) ) )
        
        elif transform.startswith("simclr"):
            _, magnitude = transform.split("_")
            transforms.append( partial(augmentations.simclr, s=float(magnitude) ) )
        
        elif transform.startswith("stacked_randaug"):
            _, _, num_layers, magnitude, s, prob = transform.split("_")
            transforms.append(partial(augmentations.stacked_randaugment,
                                      s=float(s), num_layers=int(num_layers[1:]), magnitude=int(magnitude[1:]),
                                      probability=float(prob[1:]) ))
        
        elif transform.startswith("cutout"):
            _, size = transform.split("_")
            cutout_size = int(size) // 2
            
        else:
            raise ValueError("Unknown transform : ", transform)
            
    return partial(augmentations.compose_transforms, augmentations=transforms), cutout_size

def _create_dataset(dataset_builder, split, train, transform,
                    cache=False, batch_size=128):
  ds = dataset_builder.as_dataset(split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  ds = ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  if cache:
    ds = ds.cache()

  ds = ds.prefetch(10)
  return ds


def create_split(dataset_builder, batch_size, train, dtype=tf.float32,
                 image_size=IMAGE_SIZE, cache=False, config=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.
  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
    
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    assert train_examples > config.num_labeled, "config.num_labeled should be smaller than total number of examples"
    supervised_size = config.num_labeled // jax.process_count()
    unsupervised_size = (train_examples - config.num_labeled) // jax.process_count()
    
    supervised_start = jax.process_index() * supervised_size
    unsupervised_start = jax.process_index() * unsupervised_size

    supervised_split = f'train[{supervised_start}:{supervised_start + supervised_size}]'
    unsupervised_split = f'train[{unsupervised_start}:{unsupervised_start + unsupervised_size}]'
        
  else:
    dataset_key = 'validation' if config.dataset not in ['cifar10', 'cifar100', 'svhn_cropped'] else 'test'
    validate_examples = dataset_builder.info.splits[dataset_key].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = dataset_key + f'[{start}:{start + split_size}]'


  supervised_transform, supervised_transform_cutout_size = _solve_transform(config.supervised_transform)
  unsupervised_transform_weak, unsupervised_transform_weak_cutout_size = _solve_transform(config.unsupervised_transform_weak)
  unsupervised_transform_strong, unsupervised_transform_strong_cutout_size = _solve_transform(config.unsupervised_transform_strong)

  logging.info(" supervised transform : " +  str(supervised_transform))
  logging.info(" unsupervised weak transform : " +  str(unsupervised_transform_weak))
  logging.info(" unsupervised strong transform : " +  str(unsupervised_transform_strong))
  
  unsupervised_transform = partial(two_augmented_views, first_transform=unsupervised_transform_weak,
                                   second_transform=unsupervised_transform_strong, dtype=dtype,
                                   image_size=config.image_size, config=config, 
                                   cutout_sizes=(unsupervised_transform_weak_cutout_size,
                                                 unsupervised_transform_strong_cutout_size) )

  if config.similarity_loss_on == "labeled":
    supervised_transform = partial(two_augmented_views, first_transform=supervised_transform,
                                   second_transform=supervised_transform, 
                                  image_size=config.image_size, dtype=dtype, config=config,
                                  cutout_size=(supervised_transform_cutout_size, supervised_transform_cutout_size))
  else:
    supervised_transform = partial(augmented_view, transform=supervised_transform,
                                  image_size=config.image_size, dtype=dtype, config=config,
                                  cutout_size=supervised_transform_cutout_size, no_decode=False)

  def decode_example(example, train_transform=None, single_image=False):
    if train and not single_image:
      image1, image2 = train_transform(example['image'], dtype=dtype, 
                                       image_size=config.image_size)
      
      return {"image1" : image1, "image2" : image2, "label" : example["label"]}
    elif train and single_image:
      image = train_transform(example["image"], dtype=dtype, image_size=config.image_size)
      return {"image" : image, "label" : example["label"]}
    else:
      image = preprocess_for_eval(example['image'], dtype, config.image_size, config=config)
      return {'image': image, 'label': example['label']}

  if train:
    supervised_dataset = _create_dataset(dataset_builder, supervised_split, train,
                                         partial(decode_example, train_transform=supervised_transform, single_image=config.similarity_loss_on!="labeled"), cache=cache,
                                         batch_size=batch_size)

    unsupervised_dataset = _create_dataset(dataset_builder, unsupervised_split, train,
                                           partial(decode_example, train_transform=unsupervised_transform), cache=cache,
                                           batch_size=batch_size*config.mu)
    return supervised_dataset, unsupervised_dataset
  else:
    return _create_dataset(dataset_builder, split, train, decode_example, cache=cache, batch_size=batch_size)
