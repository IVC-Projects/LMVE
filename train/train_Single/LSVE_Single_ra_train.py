import argparse, time
from LSVE_ra_model import model_single
from UTILS_single_ra import *

tf.logging.set_verbosity(tf.logging.WARN)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
EXP_DATA = 'SF_qp22_ra_01243_VDSRSingle'  # checkpoints path
HIGH_DATA_PATH = r"E:\MF\trainSet\qp22\single\high_data"  # high frames
LABEL_PATH = r"E:\MF\trainSet\qp22\single\label_s"  # lable frames
LOG_PATH = "./logs/%s/"%(EXP_DATA)
CKPT_PATH = "./checkpoints/%s/"%(EXP_DATA)
SAMPLE_PATH = "./samples/%s/"%(EXP_DATA)
PATCH_SIZE = (64, 64)
BATCH_SIZE = 64
BASE_LR = 3e-4
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 20
MAX_EPOCH = 500


parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

if __name__ == '__main__':

    # train_list return like this"[[hdata1, label1],[hdata2, label2]]" with the whole path.
    train_list = get_train_list(load_file_list(HIGH_DATA_PATH), load_file_list(LABEL_PATH))
    with tf.name_scope('input_scope'):
        train_hightData = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_gt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    shared_model = tf.make_template('shared_model', model_single)
    train_output = shared_model(train_hightData)
    train_output = tf.clip_by_value(train_output, 0., 1.)
    with tf.name_scope('loss_scope'):
        loss2 = tf.reduce_mean(tf.square(tf.subtract(train_output, train_gt)))
        loss1 = tf.reduce_sum(tf.abs(tf.subtract(train_output, train_gt)))
        W = tf.get_collection(tf.GraphKeys.WEIGHTS)
        for w in W:
            loss2 += tf.nn.l2_loss(w)*1e-4

        avg_loss = tf.placeholder('float32')
        tf.summary.scalar("avg_loss", avg_loss)

    global_step     = tf.Variable(0, trainable=False) # len(train_list)
    learning_rate   = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP*1111, LR_DECAY_RATE, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    # org ---------------------------------------------------------------------------------
    optimizer = tf.train.AdamOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(loss2, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=0)
    # org end------------------------------------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())

        if model_path:
            print("restore model...")
            saver.restore(sess, model_path)
            print("Done")

        #for epoch in range(400, MAX_EPOCH):
        for epoch in range(MAX_EPOCH):
            total_g_loss, n_iter = 0, 0
            idxOfImgs = np.random.permutation(len(train_list))

            epoch_time = time.time()

            # for idx in range(len(idxOfImgs)):
            for idx in range(1111):
                input_highData, gt_data = prepare_nn_data(train_list)
                feed_dict = {train_hightData: input_highData, train_gt: gt_data}

                _, l, output, g_step = sess.run([opt, loss2, train_output, global_step], feed_dict=feed_dict)
                total_g_loss += l
                n_iter += 1

                del input_highData, gt_data, output
            lr, summary = sess.run([learning_rate, merged], {avg_loss:total_g_loss/n_iter})
            file_writer.add_summary(summary, epoch)
            tf.logging.warning("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            print("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d.ckpt"%(EXP_DATA, epoch)))
