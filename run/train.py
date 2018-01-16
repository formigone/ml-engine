import argparse
from subprocess import call


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--job-id', type=str, required=True, help='ml-engine job ID')
  parser.add_argument('--module', type=str, required=True, help='Module to be executed')
  parser.add_argument('--model-dir', type=str, required=True, help='Name of directory within default gs bucket')

  parser.add_argument('--train-input', type=str, default='train_large_40250.tfrecords', help='Input file to train model on')
  parser.add_argument('--eval-input', type=str, default='eval_large_40250.tfrecords', help='Input file to train model on')
  parser.add_argument('--input-shape', type=str, default='flat', help='Kaggle submission file')
  parser.add_argument('--epochs', type=int, default=1, help='Amount of epochs to train for')
  parser.add_argument('--batch', type=int, default=32, help='Batch size')
  parser.add_argument('--buffer', type=int, default=128, help='Shuffle buffer size')
  parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

  return parser.parse_args()


def run(args):
  cmd = [
    'gcloud',
    'ml-engine',
    'jobs',
    'submit',
    'training',
    args.job_id,
    '--module-name', 'trainer.{}'.format(args.module),
    '--package-path', 'trainer',
    '--region', 'us-central1',
    '--job-dir', 'gs://formigone_jobs',
    '--runtime-version', '1.4',
    '--config', 'config_gpu_single.yaml',
    '--',
    '--train_input', 'gs://formigone_datasets/{}'.format(args.train_input),
    '--eval_input', 'gs://formigone_datasets/{}'.format(args.eval_input),
    '--repeat_training', str(args.epochs),
    '--model_dir', 'gs://formigone_models/{}'.format(args.model_dir),
    '--input_shape', args.input_shape,
    '--buffer_size', str(args.buffer),
    '--batch_size', str(args.batch),
    '--learning_rate', str(args.learning_rate),
    '--dropout', str(args.dropout),
  ]

  call(cmd)


if __name__ == '__main__':
  args = parse_args()
  run(args)
