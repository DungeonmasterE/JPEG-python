import cv2
import numpy as np
from bitstring import BitArray
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import time
import os
import math

import decoder
import huffmanEncode

std_luminance_quant_tbl = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61,
     12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77,
     24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101,
     72, 92, 95, 98, 112, 100, 103, 99], dtype=int)
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8, 8])

std_chrominance_quant_tbl = np.array(
    [17, 18, 24, 47, 99, 99, 99, 99,
     18, 21, 26, 66, 99, 99, 99, 99,
     24, 26, 56, 99, 99, 99, 99, 99,
     47, 66, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99], dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8, 8])

zigzagOrder = np.array(
    [0, 1, 8, 16, 9, 2, 3, 10,
     17, 24, 32, 25, 18, 11, 4, 5,
     12, 19, 26, 33, 40, 48, 41, 34,
     27, 20, 13, 6, 7, 14, 21, 28,
     35, 42, 49, 56, 57, 50, 43, 36,
     29, 22, 15, 23, 30, 37, 44, 51,
     58, 59, 52, 45, 38, 31, 39, 46,
     53, 60, 61, 54, 47, 55, 62, 63])


def padding(matrix, block_size):
    """
    将矩阵填充0，使其行数和列数都能被block_size整除

    参数:
    matrix: 输入的numpy矩阵
    block_size: 分块大小

    返回:
    填充后的矩阵
    """
    h, w, b = matrix.shape
    _h = math.ceil(h / block_size) * block_size
    _w = math.ceil(w / block_size) * block_size
    padded_matrix = np.zeros((_h, _w, b))
    padded_matrix[:h, :w, :] = matrix
    return np.uint8(padded_matrix)


# Adjust the quantization table according to the QF(q)
def light_quantization(q):
    if q < 50:
        res = np.floor((std_luminance_quant_tbl * (5000 / q) + 50) / 100)
    else:
        res = np.floor((std_luminance_quant_tbl * (200 - 2 * q) + 50) / 100)
    for i in range(8):
        for j in range(8):
            if res[i, j] > 255:
                res[i, j] = 255
            elif res[i, j] < 1:
                res[i, j] = 1
    return res


def color_quantization(q):
    if q < 50:
        res = np.rint((std_chrominance_quant_tbl * (5000 / q) + 50) / 100)
    else:
        res = np.rint((std_chrominance_quant_tbl * (200 - 2 * q) + 50) / 100)
    for i in range(8):
        for j in range(8):
            if res[i, j] > 255:
                res[i, j] = 255
            elif res[i, j] < 1:
                res[i, j] = 1
    return res


def encoding(srcFileName, q):
    img = cv2.imread(srcFileName)

    imageWidth, imageHeight, _ = img.shape

    if imageWidth % 8 != 0 or imageHeight % 8 != 0:
        img = padding(img, 8)
        _imageWidth, _imageHeight, _ = img.shape
    else:
        _imageWidth = imageWidth
        _imageHeight = imageHeight


    # split y u v
    yImageMatrix, vImageMatrix, uImageMatrix = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

    yImageMatrix = np.asarray(yImageMatrix).astype(int) - 128
    uImageMatrix = np.asarray(uImageMatrix).astype(int) - 128
    vImageMatrix = np.asarray(vImageMatrix).astype(int) - 128

    blockSum = _imageWidth // 8 * _imageHeight // 8

    sosBitStream = BitArray()

    yDC = np.zeros([blockSum], dtype=int)
    uDC = np.zeros([blockSum], dtype=int)
    vDC = np.zeros([blockSum], dtype=int)

    dyDC = np.zeros([blockSum], dtype=int)  # prediction error
    duDC = np.zeros([blockSum], dtype=int)
    dvDC = np.zeros([blockSum], dtype=int)

    yAC = np.zeros([blockSum, 63], dtype=int)
    uAC = np.zeros([blockSum, 63], dtype=int)
    vAC = np.zeros([blockSum, 63], dtype=int)

    blockNum = 0
    for y in range(0, _imageWidth, 8):
        for x in range(0, _imageHeight, 8):

            # DCT
            yDctMatrix = cv2.dct(yImageMatrix[y:y + 8, x:x + 8].astype('float32'))
            uDctMatrix = cv2.dct(uImageMatrix[y:y + 8, x:x + 8].astype('float32'))
            vDctMatrix = cv2.dct(vImageMatrix[y:y + 8, x:x + 8].astype('float32'))

            # Quantization
            yQuantMatrix = np.rint(yDctMatrix / light_quantization(q))
            uQuantMatrix = np.rint(uDctMatrix / color_quantization(q))
            vQuantMatrix = np.rint(vDctMatrix / color_quantization(q))

            # ZigZag
            yZ_Code = yQuantMatrix.reshape([64])[zigzagOrder]
            uZ_Code = uQuantMatrix.reshape([64])[zigzagOrder]
            vZ_Code = vQuantMatrix.reshape([64])[zigzagOrder]

            yDC[blockNum] = yZ_Code[0]
            yAC[blockNum] = yZ_Code[1:]
            uDC[blockNum] = uZ_Code[0]
            uAC[blockNum] = uZ_Code[1:]
            vDC[blockNum] = vZ_Code[0]
            vAC[blockNum] = vZ_Code[1:]

            if blockNum == 0:
                dyDC[blockNum] = yDC[blockNum]
                duDC[blockNum] = uDC[blockNum]
                dvDC[blockNum] = vDC[blockNum]
            else:
                dyDC[blockNum] = yDC[blockNum] - yDC[blockNum - 1]
                duDC[blockNum] = uDC[blockNum] - uDC[blockNum - 1]
                dvDC[blockNum] = vDC[blockNum] - vDC[blockNum - 1]

            yDC_huf_encode = huffmanEncode.encodeDCToBoolList(dyDC[blockNum], 1)
            sosBitStream.append(yDC_huf_encode)

            huffmanEncode.encodeACBlock(sosBitStream, yAC[blockNum], 1)

            uDC_huf_encode = huffmanEncode.encodeDCToBoolList(duDC[blockNum], 0)
            sosBitStream.append(uDC_huf_encode)

            huffmanEncode.encodeACBlock(sosBitStream, uAC[blockNum], 0)  # uAC

            vDC_huf_encode = huffmanEncode.encodeDCToBoolList(dvDC[blockNum], 0)
            sosBitStream.append(vDC_huf_encode)

            huffmanEncode.encodeACBlock(sosBitStream, vAC[blockNum], 0)

            blockNum = blockNum + 1


    # encapsulate jpg files
    jpegFile = open(enc_pic, 'wb+')

    '''
    write jpeg header
    FFD8 FFE0 0010 4A46494600 0101 00 0001 0001 00 00
    FFD8: 起始标记Start of Image(SOI)
    FFE0: 应用标记APP0, FFE0后面的数据均为具体的APP0段信息
    0010: APP0长度, 为0x10(16)个字节
    4A46494600: 文件标识符，为固定值
    0101: 版本号, 第1个数字为主版本, 第2个为次版本, 这里版本是v1.1
    后面的依次表示：像素密度单位(0表示无单位)、水平方向密度、垂直方向密度、缩略图宽度(0表示无缩略图)、缩略图高度
    '''
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))

    '''
    写入量化表
    FFDB 0043 0 0
    FFDB: 量化表标识符
    0043: 量化表长度, 为16x4+3=67个字节
    后面的依次表示：量化表位深(0为8位, 1为16位)、量化表id
    '''
    # Write luminance quantization table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = light_quantization(q).reshape([64])[zigzagOrder]
    jpegFile.write(bytes(np.uint8(luminanceQuantTbl).tolist()))

    # Write chroma quantization table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = color_quantization(q).reshape([64])[zigzagOrder]
    jpegFile.write(bytes(np.uint8(chrominanceQuantTbl).tolist()))

    '''
    Start Of Frame(SOF0)
    FFC0 0011 08
    FFC0: SOF0标识符
    0011: SOF0长度, 为17字节
    08: 精度为8位
    '''
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(imageHeight)[2:]
    wHex = hex(imageWidth)[2:]

    # 不足4位向前补0
    while len(hHex) != 4:
        hHex = '0' + hHex
    while len(wHex) != 4:
        wHex = '0' + wHex
    jpegFile.write(huffmanEncode.hexToBytes(wHex))
    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    '''
    写入颜色分量信息
    03 011100 021101 031101
    03: 为固定值, 表示YCbCr颜色空间
    011100: 颜色分量id(01为Y、02为Cb、03为Cr), 水平/垂直采样间隔(11表示水平1垂直1), 量化表id(00)
    下面为不同采样方式的编码方案：
    1:1全采样：	01 11 00	02 11 01	03 11 01
    1:2降采样：	01 21 00	02 11 01	03 11 01
    1:4降采样：	01 22 00	02 11 01	03 11 01
    '''
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    # write huffman table
    jpegFile.write(huffmanEncode.hexToBytes(

        # FFC4为标志位，01A2表示总长度为418字节
        'FFC401A2'

        # 亮度DC表（对应DCLuminanceSizeToCode），表头为00
        # {00 00 07 01 01 01 01 01 00 00 00 00 00 00 00 00}表示各码长的符号数量
        # {04 05 03 02 06 01 00 07 08 09 0A 0B}表示符号值
        '00'
        '00000701010101010000000000000000'
        '040503020601000708090A0B'

        # 色度DC表（对应DCChrominanceSizeToCode），表头为01
        # {01 00 02 02 03 01 01 01 01 01 00 00 00 00 00 00}
        # {00 01 00 02 03 04 05 06 07 08 09 0A 0B}
        '01'
        '00020203010101010100000000000000'
        '010002030405060708090A0B'

        # 亮度AC表，表头为10
        # {00 02 01 03 03 02 04 02 06 07 03 04 02 06 02 73}
        '10'
        '00020103030204020607030402060273'
        '010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'

        # 色度AC表，表头为11
        # {00 02 02 01 02 03 05 05 04 05 06 04 08 03 03 6D}
        '11'
        '0002020102030505040506040803036D'
        '0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

    # Start of Scan (SOS)
    sosLength = sosBitStream.__len__()
    filledNum = 8 - sosLength % 8
    if filledNum != 0:
        sosBitStream.append(np.ones([filledNum]).tolist())

    # FF DA 000C 03 0100 0211 0311 00 3F 00
    jpegFile.write(huffmanEncode.hexToBytes('FFDA000C03010002110311003F00'))

    # write encoded data
    sosBytes = sosBitStream.tobytes()
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]]))
        if sosBytes[i] == 255:
            jpegFile.write(bytes([0]))  # FF to FF00

    # write End Symbol
    jpegFile.write(bytes([255, 217]))  # FF D9
    jpegFile.close()


if __name__ == '__main__':
    path = 'Kagari.bmp'

    # 准备绘图数据
    _Q = []  # 质量因子
    _time = []  # 编码时间
    _decom_time = []  # 解码时间
    _com_rate = []  # 压缩率
    _PSNR = []  # 峰值信噪比
    _SSIM = []  # 结构化相似指数

    for i in range(1):
        Q = 90
        # Q = 100 - 10 * i
        _Q.append(Q)

        # 图片存储路径
        ori_pic = path
        enc_pic = "./data/" + path[:-4] + "_QF=" + str(Q) + ".jpg"
        dec_pic = "./data/" + path[:-4] + "_QF=" + str(Q) + ".bmp"

        print("\n当质量因子为" + str(Q) + "时：")
        print("----正在编码-----")
        time_start = time.time()  # 记录算法开始时间
        encoding(path, Q)
        time_end = time.time()  # 记录算法结束时间
        tim = time_end - time_start
        tim = '{:.2f}'.format(tim)  # 保留两位小数
        _time.append(tim)
        print("编码过程的执行时间为：", tim)

        # 计算压缩率
        size1 = os.path.getsize(ori_pic)
        size2 = os.path.getsize(enc_pic)
        ys_rate = size2 / size1 * 100
        ys_rate = '{:.2f}'.format(ys_rate)  # 保留两位小数
        _com_rate.append(ys_rate)
        print('压缩率为：' + str(ys_rate) + "%")

        print("-----正在解码-----")
        image = decoder.JPEG_dec().decode(enc_pic)
        cv2.imwrite(dec_pic, image)

        # 计算峰值信噪比、SSIM、图像差值
        img_original = cv2.imread(path)
        img_decompress = image
        psnr = peak_signal_noise_ratio(img_original, img_decompress)  # 求峰值信噪比PSNR
        psnr = '{:.2f}'.format(psnr)  # 保留两位小数
        ssim = structural_similarity(img_original, img_decompress, multichannel=True, channel_axis=-1)
        ssim = '{:.2f}'.format(ssim)
        _PSNR.append(psnr)
        _SSIM.append(ssim)
        print("PSNR：", psnr)
        print("SSIM：", ssim)
