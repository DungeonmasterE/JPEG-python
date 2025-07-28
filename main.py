import cv2
import numpy as np
from bitstring import BitArray

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
    img = cv2.imread(srcFileName, -1)

    imageWidth, imageHeight, _ = img.shape

    # split y u v
    yImageMatrix, vImageMatrix, uImageMatrix = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

    yImageMatrix = np.asarray(yImageMatrix).astype(int)-128
    uImageMatrix = np.asarray(uImageMatrix).astype(int)-128
    vImageMatrix = np.asarray(vImageMatrix).astype(int)-128

    blockSum = imageWidth // 8 * imageHeight // 8

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
    for y in range(0, imageHeight, 8):
        for x in range(0, imageWidth, 8):

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
    JPEG header
    FFD8 FFE0 0010 4A46494600 0101 00 0001 0001 00 00
    '''
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))

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
    '''
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(imageHeight)[2:]
    wHex = hex(imageWidth)[2:]

    while len(hHex) != 4:
        hHex = '0' + hHex
    while len(wHex) != 4:
        wHex = '0' + wHex
    jpegFile.write(huffmanEncode.hexToBytes(hHex))
    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    '''
    Write color component information
    03 011100 021101 031101
    '''
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    # write huffman table
    jpegFile.write(huffmanEncode.hexToBytes(
        'FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

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
    path = 'Lenna_RGB.bmp'
    Q = 100

    # Storage path
    enc_pic = "./data/enc_" + path[:-4] + "_QF=" + str(Q) + ".jpg"
    dec_pic = "./data/dec_" + path[:-4] + "_QF=" + str(Q) + ".bmp"

    encoding(path, Q)
    image = decoder.JPEG_dec().decode(enc_pic)

    cv2.imwrite(dec_pic, image)