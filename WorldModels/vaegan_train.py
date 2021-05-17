import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from vaegan import enc, dec, dis, train_step_vaegan
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from utils import PARSER
args = PARSER.parse_args()

def ds_gen():
    dirname = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                img_i = img / 255.0
                yield img_i

DEPTH = 32
LATENT_DEPTH = 32
K_SIZE = 5
IM_DIM = 64
batch_size = 64
inner_loss_coef = 1
normal_coef = 0.1
kl_coef = 0.01
step = 0
epoch = 0
max_step = 10000000
save_freq,save_number_mult = 1000, 10000
metrics_names = ["gan_loss", "vae_loss", "fake_dis_loss", "r_dis_loss", "t_dis_loss", "vae_inner_loss", "E_loss", "D_loss", "kl_loss", "normal_loss"]
metrics = []
for m in metrics_names: metrics.append(tf.keras.metrics.Mean('m', dtype=tf.float32))

train_log_dir = ('logs/sep_D%dL%d/' % (DEPTH,LATENT_DEPTH)) 
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
name = ('sep_D%dL%d' % (DEPTH,LATENT_DEPTH))


def log_results() :
    temp = ""
    for name,metric in zip(metrics_names,metrics): temp+= " " + name + " " + str(np.around(metric.result().numpy(), 3)) 
    print(f"\rEpoch : " + str(epoch) +" Step : " + str(step) + " " + temp, end="", flush=True)
    with train_summary_writer.as_default():
        for name,metric in zip(metrics_names,metrics): tf.summary.scalar(name, metric.result(), step=step)
    for metric in metrics: metric.reset_states()


shuffle_size = 20 * 1000 # only loads ~20 episodes for shuffle windows b/c im poor and don't have much RAM
x_train = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
x_train = x_train.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(64)
x_train = x_train.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)
encoder, decoder, discriminator = enc(), dec(), dis() 
encoder_opt, generator_opt, discriminator_opt = keras.optimizers.Adam(lr=lr),  keras.optimizers.Adam(lr=lr), keras.optimizers.Adam(lr=lr)
for i in range(15):
    epoch += 1
    for x in x_train:
        if not step % 5: log_results()
        results = train_step_vaegan(x, encoder, decoder, discriminator, encoder_opt, generator_opt, discriminator_opt, inner_loss_coef, normal_coef, kl_coef)
        for metric,result in zip(metrics, results): metric(result)
        step += 1
    encoder.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/enc.h5')
    decoder.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/dec.h5')
    discriminator.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/dis.h5')