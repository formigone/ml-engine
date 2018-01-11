import argparse
from subprocess import call


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--module', type=str, required=True, help='Module to be executed')
  parser.add_argument('--model-dir', type=str, required=True, help='Name of directory within default gs bucket')
  parser.add_argument('--output-file', type=str, required=True, help='Kaggle submission file')
  parser.add_argument('--input-shape', type=str, default='stack', help='Kaggle submission file')
  parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

  return parser.parse_args()


def run(args):
  cmd = [
    'python',
    '-m',
    'trainer.{}'.format(args.module),
    '--model_dir', args.model_dir,
    '--output_file', args.output_file,
    '--dropout', str(args.dropout),
    '--mode', 'predict',
    '--input_shape', args.input_shape,
    '--predict_input_dir', '/Users/rsilveira/rnd/tensorflow-speech-recognition-challenge/data_speech_commands_v0.01/test/audio',
    '--predict_input', '/Users/rsilveira/rnd/tensorflow-speech-recognition-challenge/gtest_alphanum.tfrecords',
  ]

  call(cmd)


if __name__ == '__main__':
  args = parse_args()
  print(args)
  run(args)
