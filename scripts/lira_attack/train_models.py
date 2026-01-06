#!/usr/bin/env python3
"""
Script to train shadow models for membership inference attacks.

This script trains multiple shadow models with different experiment IDs.
Each experiment ID corresponds to a different subset of the training data.

Usage:
    python train_shadow_models.py --num_experiments 16 --expid_list 0,15 --epochs 100 --dataset_size 50000
"""

import os
import sys
import argparse
import shutil
import json
import time

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

import numpy as np
import tensorflow as tf
from objax.util import EasyDict
from absl import flags

from train import get_data, network, MemModule

# Define flags that train.py uses (required for imports)
flags.DEFINE_string('arch', 'cnn32-3-mean', 'Model architecture.')
flags.DEFINE_float('lr', 0.1, 'Learning rate.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
flags.DEFINE_integer('batch', 256, 'Batch size')
flags.DEFINE_integer('epochs', 501, 'Training duration in number of epochs.')
flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
flags.DEFINE_integer('seed', None, 'Training seed.')
flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
flags.DEFINE_integer('expid', None, 'Experiment ID')
flags.DEFINE_integer('num_experiments', None, 'Number of experiments')
flags.DEFINE_string('augment', 'weak', 'Strong or weak augmentation')
flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')
flags.DEFINE_integer('dataset_size', 50000, 'number of examples to keep.')
flags.DEFINE_integer('eval_steps', 1, 'how often to get eval accuracy.')
flags.DEFINE_integer('abort_after_epoch', None, 'stop trainin early at an epoch')
flags.DEFINE_integer('save_steps', 10, 'how often to get save model.')
flags.DEFINE_integer('patience', None, 'Early stopping after this many epochs without progress')
flags.DEFINE_bool('tunename', False, 'Use tune name?')

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


def parse_expid_list(expid_str):
    """
    Parse expid list string into a list of integers.
    
    Supports formats:
    - "0,15" -> range(0, 16) = [0, 1, 2, ..., 15]
    - "0,1,2,3" -> [0, 1, 2, 3]
    - "0-15" -> range(0, 16) = [0, 1, 2, ..., 15]
    
    Args:
        expid_str: String representation of expid list
        
    Returns:
        List of experiment IDs
    """
    expid_str = expid_str.strip()
    
    # Handle range format "0,15" or "0-15"
    if ',' in expid_str:
        parts = expid_str.split(',')
        if len(parts) == 2:
            # Range format: "0,15" means 0 to 15 inclusive
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        else:
            # Comma-separated list: "0,1,2,3"
            return [int(x.strip()) for x in parts]
    elif '-' in expid_str:
        # Dash format: "0-15"
        parts = expid_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    else:
        # Single number
        return [int(expid_str)]


def train_single_model(expid, num_experiments, epochs, dataset_size, 
                       dataset='cifar10', arch='wrn28-2', batch=256, 
                       lr=0.1, weight_decay=0.0005, augment='weak', 
                       pkeep=0.5, save_steps=20, eval_steps=1, 
                       patience=None, only_subset=None, seed=None, 
                       data_dir=None, logs_dir=None):
    """
    Train a single shadow model with the given experiment ID.
    
    Args:
        expid: Experiment ID (determines which subset of data to use)
        num_experiments: Total number of experiments in the family
        epochs: Number of training epochs
        dataset_size: Number of examples to keep from the dataset
        dataset: Dataset name (default: 'cifar10')
        arch: Model architecture (default: 'wrn28-2')
        batch: Batch size (default: 256)
        lr: Learning rate (default: 0.1)
        weight_decay: Weight decay (default: 0.0005)
        augment: Augmentation type (default: 'weak')
        pkeep: Probability to keep examples (default: 0.5)
        save_steps: How often to save model (default: 20)
        eval_steps: How often to evaluate (default: 1)
        patience: Early stopping patience (default: None)
        only_subset: Only train on a subset of images (default: None)
        seed: Random seed (default: None, will be auto-generated)
        data_dir: Data directory (default: None, uses project_dir/data)
        logs_dir: Logs directory (default: None, uses project_dir/logs)
    """
    # Set up directories
    if data_dir is None:
        data_dir = os.path.join(project_dir, 'data')
    if logs_dir is None:
        logs_dir = os.path.join(project_dir, 'logs')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Disable GPU for TensorFlow (JAX will handle GPU)
    tf.config.experimental.set_visible_devices([], "GPU")
    
    # Set seed
    if seed is None:
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())
    
    # Create args dictionary
    args = EasyDict(
        arch=arch,
        lr=lr,
        batch=batch,
        weight_decay=weight_decay,
        augment=augment,
        seed=seed
    )
    
    # Set up log directory
    base_logdir = os.path.join(logs_dir, 'exp', dataset)
    os.makedirs(base_logdir, exist_ok=True)
    
    logdir_path = f"experiment-{expid}_{num_experiments}"
    logdir_path = os.path.join(base_logdir, logdir_path)
    
    # Check if already completed
    if os.path.exists(os.path.join(logdir_path, "ckpt", f"{epochs:010d}.npz")):
        print(f"Run {expid} already completed. Skipping.")
        return
    
    # Delete incomplete run if exists
    if os.path.exists(logdir_path):
        print(f"Deleting run {expid} that did not complete.")
        shutil.rmtree(logdir_path)
    
    print(f"Creating experiment directory: {logdir_path}")
    os.makedirs(logdir_path, exist_ok=True)
    
    # Create configuration dictionary for get_data
    config = {
        'logdir': logs_dir,
        'dataset': dataset,
        'dataset_size': dataset_size,
        'num_experiments': num_experiments,
        'expid': expid,
        'pkeep': pkeep,
        'only_subset': only_subset,
        'augment': augment,
        'batch': batch,
        'data_dir': data_dir
    }
    
    # Get data
    print(f"Loading data for experiment {expid}...")
    train_data, test_data, xs, ys, keep, nclass = get_data(seed, config)
    
    # Define the network and training module
    print(f"Initializing model for experiment {expid}...")
    tm = MemModule(
        network(arch),
        nclass=nclass,
        mnist=(dataset == 'mnist'),
        epochs=epochs,
        expid=expid,
        num_experiments=num_experiments,
        pkeep=pkeep,
        save_steps=save_steps,
        only_subset=only_subset,
        **args
    )
    
    # Save hyperparameters
    params = {}
    params.update(tm.params)
    
    with open(os.path.join(logdir_path, 'hparams.json'), 'w') as f:
        json.dump(params, f, indent=2)
    np.save(os.path.join(logdir_path, 'keep.npy'), keep)
    
    # Train
    print("=" * 80)
    print(f"Training experiment {expid}/{num_experiments-1}")
    print("=" * 80)
    tm.train(epochs, len(xs), train_data, test_data, logdir_path,
             save_steps=save_steps, patience=patience, eval_steps=eval_steps)
    
    print("=" * 80)
    print(f"✅ Training completed! Results saved to {logdir_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train shadow models for membership inference attacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train experiments 0-15 (16 models total)
  python train_shadow_models.py --num_experiments 16 --expid_list 0,15 --epochs 100 --dataset_size 50000
  
  # Train specific experiments
  python train_shadow_models.py --num_experiments 16 --expid_list 0,1,2,3 --epochs 100 --dataset_size 50000
  
  # Train with custom parameters
  python train_shadow_models.py --num_experiments 16 --expid_list 0,15 --epochs 200 --dataset_size 10000 --batch 128
        """
    )
    
    parser.add_argument('--num_experiments', type=int, required=True,
                       help='Total number of experiments in the family (e.g., 16)')
    parser.add_argument('--expid_list', type=str, required=True,
                       help='Experiment IDs to train. Format: "0,15" (range) or "0,1,2,3" (list)')
    parser.add_argument('--epochs', type=int, required=True,
                       help='Number of training epochs')
    parser.add_argument('--dataset_size', type=int, required=True,
                       help='Number of examples to keep from the dataset')
    
    # Optional parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--arch', type=str, default='wrn28-2',
                       help='Model architecture (default: wrn28-2)')
    parser.add_argument('--batch', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay (default: 0.0005)')
    parser.add_argument('--augment', type=str, default='weak',
                       choices=['weak', 'mirror', 'none'],
                       help='Augmentation type (default: weak)')
    parser.add_argument('--pkeep', type=float, default=0.5,
                       help='Probability to keep examples (default: 0.5)')
    parser.add_argument('--save_steps', type=int, default=20,
                       help='How often to save model (default: 20)')
    parser.add_argument('--eval_steps', type=int, default=1,
                       help='How often to evaluate (default: 1)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (default: None)')
    parser.add_argument('--only_subset', type=int, default=None,
                       help='Only train on a subset of images (default: None)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: None, auto-generated)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: project_dir/data)')
    parser.add_argument('--logs_dir', type=str, default=None,
                       help='Logs directory (default: project_dir/logs)')
    
    args = parser.parse_args()
    
    # Parse expid list
    expid_list = parse_expid_list(args.expid_list)
    
    print(f"Training {len(expid_list)} shadow models")
    print(f"Experiment IDs: {expid_list}")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Architecture: {args.arch}")
    print(f"Batch size: {args.batch}")
    print()
    
    # Train each model
    for i, expid in enumerate(expid_list):
        print(f"\n{'='*80}")
        print(f"Training model {i+1}/{len(expid_list)}: expid={expid}")
        print(f"{'='*80}\n")
        
        try:
            train_single_model(
                expid=expid,
                num_experiments=args.num_experiments,
                epochs=args.epochs,
                dataset_size=args.dataset_size,
                dataset=args.dataset,
                arch=args.arch,
                batch=args.batch,
                lr=args.lr,
                weight_decay=args.weight_decay,
                augment=args.augment,
                pkeep=args.pkeep,
                save_steps=args.save_steps,
                eval_steps=args.eval_steps,
                patience=args.patience,
                only_subset=args.only_subset,
                seed=args.seed,
                data_dir=args.data_dir,
                logs_dir=args.logs_dir
            )
        except Exception as e:
            print(f"❌ Error training experiment {expid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Completed training {len(expid_list)} shadow models")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

