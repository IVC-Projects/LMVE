import time
from LSVE_ra_model import model_single
from UTILS_MF_ra_ALL_Single import *
import itertools

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = "SF_qp27_ra_01232_LSVE"  # single checkpoints
TESTOUT_PATH = "./testout/%s/"%(EXP_DATA)
MODEL_PATH = "./checkpoints/%s/"%(EXP_DATA)
QP_LOWDATA_PATH = r'E:\MF\HEVC_TestSequenct\rec\qp27\ra\D\BasketballPass_qp22_ai_416x240'  # low frames
GT_PATH = r"E:\MF\HEVC_TestSequenct\org\D\BasketballPass_416x240" # ground truth
DL_path = 'E:\MF\HEVC_TestSequenct\DL_rec\Single\qp27'
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA)
NOFILTER = {'q22':42.2758, 'q27':38.9788, 'qp32':35.8667, 'q37':32.8257,'qp37':32.8257}

#  Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##cb, cr components are not implemented
def prepare_test_data(fileOrDir):
    if not os.path.exists(TESTOUT_PATH):
        os.mkdir(TESTOUT_PATH)

    singleData_ycbcr = []
    singleGT_y = []
    fileName_list = []

    #The input is a single file.
    if len(fileOrDir) == 2:
        # return the whole absolute path.
        fileName_list = load_file_list(fileOrDir[0])
        # double_list # [[high, low1, label1], [[h21,h22], low2, label2]]
        # single_list # [[low1, lable1], [2,2] ....]
        single_list = get_test_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))

        # single_list # [[low1, lable1], [2,2] ....]
        for pair in single_list:
            lowData_list = []
            lowData_imgY = c_getYdata(pair[0])
            CbCr = c_getCbCr(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            lowData_imgY = normalize(lowData_imgY)

            lowData_imgY = np.resize(lowData_imgY, (1, lowData_imgY.shape[0], lowData_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            lowData_list.append([lowData_imgY, CbCr])
            singleData_ycbcr.append(lowData_list)
            singleGT_y.append(gt_imgY)

    else:
        print("Invalid Inputs...!tjc!")
        exit(0)

    return singleData_ycbcr, singleGT_y, fileName_list

def test_all_ckpt(modelPath, fileOrDir):
    max = [0, 0]
    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)

    with tf.Session() as sess:
        lowData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model_single)
        output_tensor = shared_model(lowData_tensor)
        output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
        output_tensor = output_tensor * 255

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(OUT_DATA_PATH, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        singleData_ycbcr, singleGT_y, fileName_list = prepare_test_data(fileOrDir)

        for ckpt in ckptFiles:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            if epoch != 169:
                continue

            saver.restore(sess, os.path.join(modelPath,ckpt))
            total_time, total_psnr = 0, 0
            total_imgs = len(fileName_list)
            count = 0
            for i in range(total_imgs):
                if i%4==0:
                    continue
                count += 1

                # sorry! this place write so difficult!【[[[h1,0]],[[low,0]],[[h2, 0]]], [[[h1,0]],[[low,0]],[[h2, 0]]]】
                lowDataY = singleData_ycbcr[i][0][0]
                imgLowCbCr = singleData_ycbcr[i][0][1]
                gtY = singleGT_y[i] if singleGT_y else 0

                #### adopt the split frame method to deal with the out of memory situation. ####

                start_t = time.time()
                out = sess.run(output_tensor, feed_dict={lowData_tensor: lowDataY})
                out = np.around(out)
                out = out.astype('int')
                out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])
                hevc = psnr(lowDataY * 255.0, gtY)
                duration_t = time.time() - start_t
                total_time += duration_t
                Y = np.reshape(out, [out.shape[1], out.shape[2]])
                Y = np.array(list(itertools.chain.from_iterable(Y)))
                U = imgLowCbCr[0]
                V = imgLowCbCr[1]
                creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                if not os.path.exists(creatPath):
                    os.mkdir(creatPath)

                if singleGT_y:
                    p = psnr(out, gtY)
                    path = os.path.join(DL_path,
                                        fileName_list[i].split('\\')[-2],
                                        fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p - hevc) + '.yuv'

                    YUV = np.concatenate((Y, U, V))
                    YUV = YUV.astype('uint8')
                    YUV.tofile(path)

                    total_psnr += p
                    print("qp??\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))
                #print("took:%.2fs\t psnr:%.2f name:%s"%(duration_t, p, save_path))



            print("AVG_DURATION:%.2f\tAVG_PSNR:%.4f"%(total_time/total_imgs, total_psnr / count))
            print('count:', count)
            # avg_psnr = total_psnr/total_imgs
            avg_psnr = total_psnr / count
            avg_duration = (total_time/total_imgs)
            if avg_psnr > max[0]:
                max[0] = avg_psnr
                max[1] = epoch

if __name__ == '__main__':
    test_all_ckpt(MODEL_PATH, [QP_LOWDATA_PATH, GT_PATH])