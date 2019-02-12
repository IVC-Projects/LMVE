import time
import itertools
from LMVE_model import model_double
from LSVE_model import model_single
from UTILS_MF_ra import *

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA1 = "MF_qp22_ra_01222_LMVE_double"  # double checkpoints
EXP_DATA2 = "MF_qp22_ra_01222_LMVE_double"  # single checkpoints   not be used
MODEL_DOUBLE_PATH1 = "./checkpoints/%s/"%(EXP_DATA1)
MODEL_DOUBLE_PATH2 = "./checkpoints/%s/"%(EXP_DATA2)
HIGHDATA_Parent_PATH = r"E:\MF\HEVC_TestSequenct\qp37_ldp\warped_yuv\FourPeople_qp22_ai_1280x720"  # warped high frames
QP_LOWDATA_PATH = r'E:\MF\HEVC_TestSequenct\rec\qp37\ldp\E\FourPeople_qp37_ldp_1280x720'  # low frames
GT_PATH = r"E:\MF\HEVC_TestSequenct\org\E\FourPeople_1280x720"  # ground truth
DL_path = 'E:\MF\HEVC_TestSequenct\DL_rec\double\qp37'
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA1)


#  Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##cb, cr components are not implemented
def prepare_test_data(fileOrDir):
    doubleData_ycbcr = []
    doubleGT_y = []
    singleData_ycbcr = []
    singleGT_y = []
    fileName_list = []
    #The input is a single file.
    if len(fileOrDir) == 3:
        # return the whole absolute path.
        fileName_list = load_file_list(fileOrDir[1])
        # double_list # [[high, low1, label1], [[h21,h22], low2, label2]]
        # single_list # [[low1, lable1], [2,2] ....]
        double_list, single_list = get_test_list(HIGHDATA_Parent_PATH, load_file_list(fileOrDir[1]),
                                   load_file_list(fileOrDir[2]))

        for pair in double_list:
            high1Data_List = []
            lowData_List = []
            high2Data_List = []

            high1Data_imgY = c_getYdata(pair[0][0])
            high2Data_imgY = c_getYdata(pair[0][1])
            lowData_imgY = c_getYdata(pair[1])
            CbCr = c_getCbCr(pair[1])
            gt_imgY = c_getYdata(pair[2])

            #normalize
            high1Data_imgY = normalize(high1Data_imgY)
            lowData_imgY = normalize(lowData_imgY)
            high2Data_imgY = normalize(high2Data_imgY)

            high1Data_imgY = np.resize(high1Data_imgY, (1, high1Data_imgY.shape[0], high1Data_imgY.shape[1],1))
            lowData_imgY = np.resize(lowData_imgY, (1, lowData_imgY.shape[0], lowData_imgY.shape[1], 1))
            high2Data_imgY = np.resize(high2Data_imgY, (1, high2Data_imgY.shape[0], high2Data_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1],1))

            ## act as a placeholder

            high1Data_List.append([high1Data_imgY, 0])
            lowData_List.append([lowData_imgY, CbCr])
            high2Data_List.append([high2Data_imgY, 0])
            doubleData_ycbcr.append([high1Data_List, lowData_List, high2Data_List])
            doubleGT_y.append(gt_imgY)

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

    return doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list

def test_all_ckpt(modelPath1, modelPath2, fileOrDir):
    max = [0, 0]
    tem1 = [f for f in os.listdir(modelPath1) if 'data' in f]
    ckptFiles1 = sorted([r.split('.data')[0] for r in tem1])
    tem2 = [f for f in os.listdir(modelPath2) if 'data' in f]
    ckptFiles2 = sorted([r.split('.data')[0] for r in tem2])
    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)

    doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list = prepare_test_data(fileOrDir)
    total_time, total_psnr = 0, 0
    total_imgs = len(fileName_list)
    count = 0
    for i in range(total_imgs):
        if i % 4 != 0:
            count += 1
            # sorry! this place write so difficult!【[[[h1,0]],[[low,0]],[[h2, 0]]], [[[h1,0]],[[low,0]],[[h2, 0]]]】
            j = i - (i//4) - 1
            imgHigh1DataY = doubleData_ycbcr[j][0][0][0]
            imgLowDataY = doubleData_ycbcr[j][1][0][0]
            imgLowCbCr = doubleData_ycbcr[j][1][0][1]
            imgHigh2DataY = doubleData_ycbcr[j][2][0][0]
            gtY = doubleGT_y[j] if doubleGT_y else 0
            start_t = time.time()
            for ckpt1 in ckptFiles1:
                epoch = int(ckpt1.split('_')[-1].split('.')[0])
                if epoch != 149:
                    continue
                # very important!!!!!!!
                tf.reset_default_graph()
                # Double section
                high1Data_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                lowData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                high2Data_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                shared_model1 = tf.make_template('shared_model', model_double)
                output_tensor1 = shared_model1(high1Data_tensor, lowData_tensor, high2Data_tensor)
                # output_tensor = shared_model(input_tensor)
                output_tensor1 = tf.clip_by_value(output_tensor1, 0., 1.)
                output_tensor1 = output_tensor1 * 255
                with tf.Session() as sess:
                    saver = tf.train.Saver(tf.global_variables())
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath1, ckpt1))
                    out = sess.run(output_tensor1, feed_dict={high1Data_tensor: imgHigh1DataY,
                                                             lowData_tensor:imgLowDataY, high2Data_tensor: imgHigh2DataY})
                    hevc = psnr(imgLowDataY * 255.0, gtY)
                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])

                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)

                    if doubleGT_y:
                        p = psnr(out, gtY)

                        path = os.path.join(DL_path,
                                            fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p-hevc)+ '.yuv'

                        YUV = np.concatenate((Y, U, V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)

                        total_psnr += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

            duration_t = time.time() - start_t
            total_time += duration_t
        else:   #single frame 
            continue
            count += 1
            j = i // 4
            lowDataY = singleData_ycbcr[j][0][0]
            imgLowCbCr = singleData_ycbcr[j][0][1]
            gtY = singleGT_y[j] if singleGT_y else 0
            hevc = psnr(lowDataY * 255.0, gtY)
            start_t = time.time()
            for ckpt2 in ckptFiles2:
                epoch = int(ckpt2.split('_')[-1].split('.')[0])
                if epoch != 169:
                    continue

                tf.reset_default_graph()
                # Single section
                lowSingleData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                shared_model2 = tf.make_template('shared_model', model_single)
                output_tensor2 = shared_model2(lowSingleData_tensor)
                output_tensor2 = tf.clip_by_value(output_tensor2, 0., 1.)
                output_tensor2 = output_tensor2 * 255
                with tf.Session() as sess:
                    saver = tf.train.Saver(tf.global_variables())
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath2, ckpt2))
                    out = sess.run(output_tensor2, feed_dict={lowSingleData_tensor: lowDataY})
                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])
                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)

                    if singleGT_y:
                        p = psnr(out, gtY)
                        path = os.path.join(DL_path, fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p - hevc) + '.yuv'
                        YUV = np.concatenate((Y, U, V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)

                        total_psnr += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

            duration_t = time.time() - start_t
            total_time += duration_t


    print("AVG_DURATION:%.2f\tAVG_PSNR:%.4f"%(total_time/total_imgs, total_psnr / count))
    print('count:', count)
    avg_psnr = total_psnr / count
    if avg_psnr > max[0]:
        max[0] = avg_psnr
        max[1] = epoch

if __name__ == '__main__':
    test_all_ckpt(MODEL_DOUBLE_PATH1, MODEL_DOUBLE_PATH2, [HIGHDATA_Parent_PATH, QP_LOWDATA_PATH, GT_PATH])