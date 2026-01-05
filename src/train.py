# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import functools
import os
import shutil
from typing import Callable
import json

import jax
import jax.numpy as jn
import numpy as np
import tensorflow as tf  
import tensorflow_datasets as tfds
from absl import app, flags

import objax
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet

from dataset import DataSet

FLAGS = flags.FLAGS

def augment(x, shift: int, mirror=True):
    """
    Augmentation function used in training the model.
    """
    y = x['image']
    if mirror:
        y = tf.image.random_flip_left_right(y)
    y = tf.pad(y, [[shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT')
    y = tf.image.random_crop(y, tf.shape(x['image']))
    return dict(image=y, label=x['label'])


class TrainLoop(objax.Module):
    """
    Training loop for general machine learning models.
    Based on the training loop from the objax CIFAR10 example code.
    """
    predict: Callable
    train_op: Callable

    def __init__(self, nclass: int, **kwargs):
        self.nclass = nclass
        self.params = EasyDict(kwargs)

    def train_step(self, summary: Summary, data: dict, progress: np.ndarray):
        kv = self.train_op(progress, data['image'].numpy(), data['label'].numpy())
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN, try reducing learning rate', k)
            if summary is not None:
                summary.scalar(k, float(v))

    def train(self, num_train_epochs: int, train_size: int, train: DataSet, test: DataSet, logdir: str, save_steps=100, patience=None, eval_steps=100):
        """
        Completely standard training. Nothing interesting to see here.
        """
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=20, makedir=True)
        start_epoch, last_ckpt = checkpoint.restore(self.vars())
        train_iter = iter(train)
        progress = np.zeros(jax.local_device_count(), 'f')  # for multi-GPU

        best_acc = 0
        best_acc_epoch = -1

        with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            for epoch in range(start_epoch, num_train_epochs):
                # Train
                summary = Summary()
                loop = range(0, train_size, self.params.batch)
                for step in loop:
                    progress[:] = (step + (epoch * train_size)) / (num_train_epochs * train_size)
                    self.train_step(summary, next(train_iter), progress)

                # Eval
                accuracy, total = 0, 0
                if epoch%eval_steps == 0 and test is not None:
                    for data in test:
                        total += data['image'].shape[0]
                        preds = np.argmax(self.predict(data['image'].numpy()), axis=1)
                        accuracy += (preds == data['label'].numpy()).sum()
                    accuracy /= total
                    summary.scalar('eval/accuracy', 100 * accuracy)
                    tensorboard.write(summary, step=(epoch + 1) * train_size)
                    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, summary['losses/xe'](),
                                                                    summary['eval/accuracy']()))

                    if summary['eval/accuracy']() > best_acc:
                        best_acc = summary['eval/accuracy']()
                        best_acc_epoch = epoch
                    elif patience is not None and epoch > best_acc_epoch + patience:
                        print("early stopping!")
                        checkpoint.save(self.vars(), epoch + 1)
                        return

                else:
                    print('Epoch %04d  Loss %.2f  Accuracy --' % (epoch + 1, summary['losses/xe']()))

                if epoch%save_steps == save_steps-1:
                    checkpoint.save(self.vars(), epoch + 1)


# We inherit from the training loop and define predict and train_op.
class MemModule(TrainLoop):
    def __init__(self, model: Callable, nclass: int, mnist=False, **kwargs):
        """
        Completely standard training. Nothing interesting to see here.
        """
        super().__init__(nclass, **kwargs)
        self.model = model(1 if mnist else 3, nclass)
        self.opt = objax.optimizer.Momentum(self.model.vars())
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999, debias=True)

        @objax.Function.with_vars(self.model.vars())
        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay, {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        gv = objax.GradValues(loss, self.model.vars())
        self.gv = gv

        @objax.Function.with_vars(self.vars())
        def train_op(progress, x, y):
            g, v = gv(x, y)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            lr = lr * jn.clip(progress*100,0,1)
            self.opt(lr, g)
            self.model_ema.update_ema()
            return {'monitors/lr': lr, **v[1]}

        self.predict = objax.Jit(objax.nn.Sequential([objax.ForceArgs(self.model_ema, training=False)]))

        self.train_op = objax.Jit(train_op)


def network(arch: str):
    if arch == 'cnn32-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn32-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'cnn64-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn64-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'wrn28-1':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=1)
    elif arch == 'wrn28-2':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=2)
    elif arch == 'wrn28-10':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=10)
    raise ValueError('Architecture not recognized', arch)

def get_data(seed, config_dict):
    """
    This is the function to generate subsets of the data for training models.

    First, we get the training dataset either from the numpy cache
    or otherwise we load it from tensorflow datasets.

    Then, we compute the subset. This works in one of two ways.

    1. If we have a seed, then we just randomly choose examples based on
       a prng with that seed, keeping config_dict['pkeep'] fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier.
       If we run each experiment independently then even after a lot of trials
       there will still probably be some examples that were always included
       or always excluded. So instead, with experiment IDs, we guarantee that
       after config_dict['num_experiments'] are done, each example is seen exactly half
       of the time in train, and half of the time not in train.

    Args:
        seed: Random seed for data subset selection
    """
    logdir = config_dict['logdir']
    dataset_name = config_dict['dataset']
    dataset_size = config_dict['dataset_size']
    num_experiments = config_dict['num_experiments']
    expid = config_dict['expid']
    pkeep = config_dict['pkeep']
    only_subset = config_dict['only_subset']
    augment_type = config_dict['augment']
    batch_size = config_dict['batch']
    data_dir = config_dict['data_dir']

    if os.path.exists(os.path.join(logdir, "x_train.npy")):
        inputs = np.load(os.path.join(logdir, "x_train.npy"))
        labels = np.load(os.path.join(logdir, "y_train.npy"))
    else:
        print("First time, creating dataset")
        data = tfds.as_numpy(tfds.load(name=dataset_name, batch_size=-1, data_dir=data_dir))
        inputs = data['train']['image']
        labels = data['train']['label']

        inputs = (inputs/127.5)-1
        np.save(os.path.join(logdir, "x_train.npy"), inputs)
        np.save(os.path.join(logdir, "y_train.npy"), labels)
    
    inputs = inputs[:dataset_size]
    labels = labels[:dataset_size]
    nclass = np.max(labels)+1

    np.random.seed(seed)
    if num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
        order = keep.argsort(0)
        keep = order < int(pkeep * num_experiments)
        keep = np.array(keep[expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=dataset_size) <= pkeep

    if only_subset is not None:
        keep[only_subset:] = 0

    xs = inputs[keep]
    ys = labels[keep]

    if augment_type == 'weak':
        aug = lambda x: augment(x, 4)
    elif augment_type == 'mirror':
        aug = lambda x: augment(x, 0)
    elif augment_type == 'none':
        aug = lambda x: augment(x, 0, mirror=False)
    else:
        raise ValueError(f"Unknown augmentation type: {augment_type}")

    train = DataSet.from_arrays(xs, ys, augment_fn=aug)
    test = DataSet.from_tfds(tfds.load(name=dataset_name, split='test', data_dir=DATA_DIR), xs.shape[1:])
    train = train.cache().shuffle(8192).repeat().parse().augment().batch(batch_size)
    train = train.nchw().one_hot(nclass).prefetch(16)
    test = test.cache().parse().batch(batch_size).nchw().prefetch(16)

    return train, test, xs, ys, keep, nclass