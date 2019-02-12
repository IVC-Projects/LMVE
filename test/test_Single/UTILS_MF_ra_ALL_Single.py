import numpy as np
import tensorflow as tf
import math, os, random, re
from PIL import Image
BATCH_SIZE = 64
PATCH_SIZE = (64, 64)

# due to a batch trainingSet come from one picture. I design a algorithm to make the TrainingSet more diversity.
def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)

def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)

def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input

def remap(input):
    input = 16+219/255*input
    return truncate(input, 16.0, 235.0)

def deremap(input):
    input = (input-16)*255/219
    return truncate(input, 0.0, 255.0)

# return the whole absolute path.
def load_file_list(directory):
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        list.append(os.path.join(directory,filename))
    return list

def searchHighData(currentLowDataIndex, highDataList, highIndexList):
    searchOffset = 3
    searchedHighDataIndexList = []
    searchedHighData = []
    for i in range(currentLowDataIndex - searchOffset, currentLowDataIndex + searchOffset + 1):
        if i in highIndexList:
            searchedHighDataIndexList.append(i)
    assert len(searchedHighDataIndexList) == 2, 'search method have error!'
    for tempData in highDataList:
        if int(os.path.basename(tempData).split('.')[0].split('_')[-1]) \
            == searchedHighDataIndexList[0] == searchedHighDataIndexList[1]:
            searchedHighData.append(tempData)
    return searchedHighData


#  return like this"[[[high1Data, lowData], label], [[2, 7, 8], 22], [[3, 8, 9], 33]]" with the whole path.
def get_test_list2(highDataList, lowDataList, labelList):
    assert len(lowDataList) == len(labelList), "low:%d, label:%d,"%(len(lowDataList) , len(labelList))

    # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    highIndexList = [q for q in range(49) if q % 4 == 0]
    test_list = []
    for tempDataPath in lowDataList:
        tempData = []
        temp = []
        # this place should changed on the different situation.
        currentLowDataIndex = int(os.path.basename(tempDataPath).split('.')[0].split('_')[-1])
        searchedHighData = searchHighData(currentLowDataIndex, highDataList, highIndexList)
        tempData.append(searchedHighData[0])
        tempData.append(tempDataPath)
        tempData.append(searchedHighData[1])

        i = list(lowDataList).index(tempDataPath)

        temp.append(tempData)
        temp.append(labelList[i])

        test_list.append(temp)
    return test_list

def get_temptest_list(high1DataList, lowDataList, high2DataList, labelList):
    tempData = []
    temp = []
    test_list = []
    for i in range(len(lowDataList)):
        tempData.append(high1DataList[i])
        tempData.append(lowDataList[i])
        tempData.append(high2DataList[i])
        temp.append(tempData)
        temp.append(labelList[i])

        test_list.append(temp)
    return test_list

# [[high, low1, label1], [[h21,h22], low2, label2]]
def get_test_list(lowDataList, labelList):
    singleTest_list = []

    for lowdata in lowDataList:
        tempData = []

        tempData.append(lowdata)
        labelIndex = list(lowDataList).index(lowdata)
        if int(os.path.basename(lowdata).split('.')[0].split('_')[-1]) == \
            int(os.path.basename(labelList[labelIndex]).split('.')[0].split('_')[-1]):
            tempData.append(labelList[labelIndex])
            singleTest_list.append(tempData)

    return singleTest_list

#  return like this"[[[high1Data, lowData, high2Data], label], [[2, 7, 8], 22], [[3, 8, 9], 33]]" with the whole path.
def get_train_list(high1DataList, lowDataList, high2DataList, labelList):
    assert len(lowDataList) == len(high1DataList) == len(labelList) == len(high2DataList), \
        "low:%d, high1:%d, label:%d, high2:%d"%(len(lowDataList), len(high1DataList), len(labelList), len(high2DataList))

    train_list = []
    for i in range(len(labelList)):
        tempData = []
        temp = []
        # this place should changed on the different situation.
        if int(os.path.basename(high1DataList[i]).split('_')[-1].split('.')[0]) + 4 == \
                int(os.path.basename(lowDataList[i]).split('_')[-1].split('.')[0]) + 2 == \
                int(os.path.basename(high2DataList[i]).split('_')[-1].split('.')[0]):
            tempData.append(high1DataList[i])
            tempData.append(lowDataList[i])
            tempData.append(high2DataList[i])
            temp.append(tempData)
            temp.append(labelList[i])

        else:
            raise Exception('len(lowData) not equal with len(highData)...')
        train_list.append(temp)
    return train_list

def prepare_nn_data(train_list):
    batchSizeRandomList = random.sample(range(0,len(train_list)), 8)
    gt_list = []
    high1Data_list = []
    lowData_list = []
    high2Data_list = []
    for i in batchSizeRandomList:
        high1Data_image = c_getYdata(train_list[i][0][0])
        lowData_image = c_getYdata(train_list[i][0][1])
        high2Data_image = c_getYdata(train_list[i][0][2])
        gt_image = c_getYdata(train_list[i][1])
        for j in range(0, 8):
            #crop images to the disired size.
            high1Data_imgY, lowData_imgY, high2Data_imgY, gt_imgY = \
                crop(high1Data_image, lowData_image, high2Data_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")
            #normalize
            high1Data_imgY = normalize(high1Data_imgY)
            lowData_imgY = normalize(lowData_imgY)
            high2Data_imgY = normalize(high2Data_imgY)
            gt_imgY = normalize(gt_imgY)

            high1Data_list.append(high1Data_imgY)
            lowData_list.append(lowData_imgY)
            high2Data_list.append(high2Data_imgY)
            gt_list.append(gt_imgY)

    high1Data_list = np.resize(high1Data_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    lowData_list = np.resize(lowData_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    high2Data_list = np.resize(high2Data_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return high1Data_list, lowData_list, high2Data_list, gt_list

def getWH(yuvfileName):
    deyuv=re.compile(r'(.+?)\.')
    deyuvFilename=deyuv.findall(yuvfileName)[0] #去yuv后缀的文件名
    if 'x' in os.path.basename(deyuvFilename).split('_')[-2]:
        wxh = os.path.basename(deyuvFilename).split('_')[-2]
    elif 'x' in os.path.basename(deyuvFilename).split('_')[1]:
        wxh = os.path.basename(deyuvFilename).split('_')[1]
    else:
        raise Exception('do not find wxh')
    w, h = wxh.split('x')
    return int(w), int(h)

def c_getCbCr(path):
    w, h = getWH(path)
    CbCr = []
    with open(path, 'rb+') as file:
        y = file.read(h * w)
        if y == b'':
            return ''
        u = file.read(h * w // 4)
        v = file.read(h * w // 4)
        # convert string-list to int-list.
        u = list(map(int, u))
        v = list(map(int, v))
        CbCr.append(u)
        CbCr.append(v)
        return CbCr


def getYdata(path, size):
    w = size[0]
    h = size[1]
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)

        Yt = np.asarray(tem, dtype='float32')
    return Yt


def c_getYdata(path):
    return getYdata(path, getWH(path))

def img2y(input_img):
    if np.asarray(input_img).shape[2] == 3:
        input_imgY = input_img.convert('YCbCr').split()[0]
        input_imgCb, input_imgCr = input_img.convert('YCbCr').split()[1:3]

        input_imgY = np.asarray(input_imgY, dtype='float32')
        input_imgCb = np.asarray(input_imgCb, dtype='float32')
        input_imgCr = np.asarray(input_imgCr, dtype='float32')


        #Concatenate Cb, Cr components for easy, they are used in pair anyway.
        input_imgCb = np.expand_dims(input_imgCb,2)
        input_imgCr = np.expand_dims(input_imgCr,2)
        input_imgCbCr = np.concatenate((input_imgCb, input_imgCr), axis=2)

    elif np.asarray(input_img).shape[2] == 1:
        print("This image has one channal only.")
        #If the num of channal is 1, remain.
        input_imgY = input_img
        input_imgCbCr = None
    else:
        print("The num of channal is neither 3 nor 1.")
        exit()
    return input_imgY, input_imgCbCr

# def crop(input_image, gt_image, patch_width, patch_height, img_type):
def crop(high1Data_image, lowData_image, high2Data_image, gt_image, patch_width, patch_height, img_type):
    assert type(high1Data_image) == type(gt_image) == type(lowData_image) == type(high2Data_image), "types are different."
    high1Data_cropped = []
    lowData_cropped = []
    high2Data_cropped = []
    gt_cropped = []

    # return a ndarray object
    if img_type == "ndarray":
        in_row_ind   = random.randint(0,high1Data_image.shape[0]-patch_width)
        in_col_ind   = random.randint(0,high1Data_image.shape[1]-patch_height)

        high1Data_cropped = high1Data_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]
        lowData_cropped = lowData_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]
        high2Data_cropped = high2Data_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]

    #return an "Image" object
    elif img_type == "Image":
        pass
    return high1Data_cropped, lowData_cropped, high2Data_cropped, gt_cropped

def save_images(inputY, inputCbCr, size, image_path):
    """Save mutiple images into one single image.

    # Parameters
    # -----------
    # images : numpy array [batch, w, h, c]
    # size : list of two int, row and column number.
    #     number of images should be equal or less than size[0] * size[1]
    # image_path : string.
    #
    # Examples
    # ---------
    # # >>> images = np.random.rand(64, 100, 100, 3)
    # # >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img

    inputY = inputY.astype('uint8')
    inputCbCr = inputCbCr.astype('uint8')
    output_concat = np.concatenate((inputY, inputCbCr), axis=3)

    assert len(output_concat) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(output_concat))

    new_output = merge(output_concat, size)

    new_output = new_output.astype('uint8')

    img = Image.fromarray(new_output, mode='YCbCr')
    img = img.convert('RGB')
    img.save(image_path)

def get_image_batch(train_list,offset,batch_size):
    target_list = train_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    inputcbcr_list = []
    for pair in target_list:
        input_img = Image.open(pair[0])
        gt_img = Image.open(pair[1])

        #crop images to the disired size.
        input_img, gt_img = crop(input_img, gt_img, PATCH_SIZE[0], PATCH_SIZE[1], "Image")

        #focus on Y channal only
        input_imgY, input_imgCbCr = img2y(input_img)
        gt_imgY, gt_imgCbCr = img2y(gt_img)
        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        inputcbcr_list.append(input_imgCbCr)

    input_list = np.resize(input_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list, inputcbcr_list

def save_test_img(inputY, inputCbCr, path):
    assert len(inputY.shape) == 4, "the tensor Y's shape is %s"%inputY.shape
    assert inputY.shape[0] == 1, "the fitst component must be 1, has not been completed otherwise.{}".format(inputY.shape)

    inputY = np.squeeze(inputY, axis=0)
    inputY = inputY.astype('uint8')

    inputCbCr = inputCbCr.astype('uint8')

    output_concat = np.concatenate((inputY, inputCbCr), axis=2)
    img = Image.fromarray(output_concat, mode='YCbCr')
    img = img.convert('RGB')
    img.save(path)

def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if((type(hr_image)==type(np.array([]))) or (type(hr_image)==type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff*diff)
        mse = np.maximum(eps, mse)
        return float(10*math.log10(max_value*max_value/mse))
    else:
        assert len(hr_image.shape)==4 and len(sr_image.shape)==4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10*tf.log(max_value*max_value/mse)/math.log(10)

def getBeforeNNBlockDict(img, w, h):
    blockSize = 1000
    padding = 32
    yBlockNum = (h // blockSize) if (h % blockSize == 0) else (h // blockSize + 1)
    xBlockNum = (w // blockSize) if (w % blockSize == 0) else (w // blockSize + 1)
    tempImg = {}
    i = 0
    for yBlock in range(yBlockNum):
        for xBlock in range(xBlockNum):
            if yBlock == 0:
                if xBlock == 0:
                    tempImg[i] = img[0: blockSize+padding, 0: blockSize+padding]
                elif xBlock == xBlockNum - 1:
                    tempImg[i] = img[0: blockSize+padding, xBlock*blockSize-padding: w]
                else:
                    tempImg[i] = img[0: blockSize+padding, blockSize*xBlock-padding: blockSize*(xBlock+1)+padding]
            elif yBlock == yBlockNum - 1:
                if xBlock == 0:
                    tempImg[i] = img[blockSize*yBlock-padding: h, 0: blockSize+padding]
                elif xBlock == xBlockNum - 1:
                    tempImg[i] = img[blockSize*yBlock-padding: h, blockSize*xBlock-padding: w]
                else:
                    tempImg[i] = img[blockSize*yBlock-padding: h, blockSize*xBlock-padding: blockSize*(xBlock+1)+padding]
            elif xBlock == 0:
                tempImg[i] = img[blockSize*yBlock-padding: blockSize*(yBlock+1)+padding, 0: blockSize+padding]
            elif xBlock == xBlockNum - 1:
                tempImg[i] = img[blockSize*yBlock-padding: blockSize*(yBlock+1)+padding, blockSize*xBlock-padding: w]
            else:
                tempImg[i] = img[blockSize*yBlock-padding: blockSize*(yBlock+1)+padding,
                                 blockSize*xBlock-padding: blockSize*(xBlock+1)+padding]
            i += i
            l = tempImg[i].astype('uint8')
            l = Image.fromarray(l)
            l.show()
