from absl import flags
from absl import app
from mint.core import inputs
from mint.core import model_builder
from mint.utils import config_util
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', './checkpoints',
                    'Directory to write training checkpoints and logs')
flags.DEFINE_string('data_files', './predictions/motion-inputs/*_tfrecord-predict*',
                    'Directory to write training checkpoints and logs')
flags.DEFINE_string('config_path', './configs/fact_v5_deeper_t10_cm12.config', 'Path to the config file.')


def main(_):
  strategy = tf.distribute.MirroredStrategy()
  configs = config_util.get_configs_from_pipeline_file(FLAGS.config_path)
  model_config = configs['model']
  dataset_config = configs['eval_config']
  predict_dataset_config = configs['eval_dataset']
  model = model_builder.build(model_config, True)
  latest_ck = tf.train.latest_checkpoint(FLAGS.model_dir)
  checkpoint = tf.train.Checkpoint(model)
  checkpoint.restore(latest_ck).expect_partial()
  input = inputs.create_input(dataset_config, predict_dataset_config, use_tpu=False)
  print("\n\n\n\n\n\n\n\n")
  print(input)
  single = input.take(1)
  for element in single:
    prediction = model(element, training=False)
  print("\n\n\n\n\n\n\n\n")
  print(prediction.shape)
  #for riga in prediction:
    	#np.savetxt('./predictedValues.csv', riga, delimiter=',')

if __name__ == '__main__':
  app.run(main)