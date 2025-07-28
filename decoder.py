from struct import *
import numpy as np
import cv2

zigzag = [0, 1, 8, 16, 9, 2, 3, 10,
          17, 24, 32, 25, 18, 11, 4, 5,
          12, 19, 26, 33, 40, 48, 41, 34,
          27, 20, 13, 6, 7, 14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36,
          29, 22, 15, 23, 30, 37, 44, 51,
          58, 59, 52, 45, 38, 31, 39, 46,
          53, 60, 61, 54, 47, 55, 62, 63]


def integrate_block(height, width, blocks):
    tlist = []
    for b in blocks:
        b = np.array(b)
        tlist.append(b.reshape(8, 8))

    rlist = []
    for hi in range(height // 8):
        start = hi * width // 8
        rlist.append(np.hstack(tuple(tlist[start: start + (width // 8)])))
    matrix = np.vstack(tuple(rlist))
    return matrix


def GetArray(type, l, length):
    s = ""
    for i in range(length):
        s = s + type
    return list(unpack(s, l[:length]))


def DecodeNumber(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)


def RemoveFF00(data):
    datapro = []
    i = 0
    while True:
        b, bnext = unpack("BB", data[i:i + 2])
        if b == 0xff:
            if bnext != 0:
                break
            datapro.append(data[i])
            i += 2
        else:
            datapro.append(data[i])
            i += 1
    return datapro, i


# convert a string into a bit stream
class Stream:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def GetBit(self):
        b = self.data[self.pos >> 3]
        s = 7 - (self.pos & 0x7)
        self.pos += 1
        return (b >> s) & 1

    def GetBitN(self, l):
        val = 0
        for i in range(l):
            val = val * 2 + self.GetBit()
        return val


# Create huffman bits from table lengths
class HuffmanTable:
    def __init__(self):
        self.root = []
        self.elements = []

    def BitsFromLengths(self, root, element, pos):
        if isinstance(root, list):
            if pos == 0:
                if len(root) < 2:
                    root.append(element)
                    return True
                return False
            for i in [0, 1]:
                if len(root) == i:
                    root.append([])
                if self.BitsFromLengths(root[i], element, pos - 1):
                    return True
        return False

    def GetHuffmanBits(self, lengths, elements):
        self.elements = elements
        ii = 0
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                self.BitsFromLengths(self.root, elements[ii], i)
                ii += 1

    def Find(self, st):
        r = self.root
        while isinstance(r, list):
            r = r[st.GetBit()]
        return r

    def GetCode(self, st):
        while True:
            res = self.Find(st)
            if res == 0:
                return 0
            elif res != -1:
                return res


# main class that decodes the jpeg
class JPEG_dec:
    def __init__(self):
        self.quant = {}
        self.quantMapping = []
        self.tables = {}
        self.width = 0
        self.height = 0

    def BuildMatrix(self, st, idx, quant, olddccoeff):
        Coeff = np.zeros(64, int)
        code = self.tables[0 + idx].GetCode(st)
        bits = st.GetBitN(code)

        predict_error = int(DecodeNumber(code, bits))
        dccoeff = predict_error + olddccoeff
        dcquant = dccoeff * quant[0]
        Coeff[0] = dcquant

        l = 1
        while l < 64:
            code = self.tables[16 + idx].GetCode(st)
            if code == 0:
                break
            if code > 15:
                l += (code >> 4)
                code = code & 0xf

            bits = st.GetBitN(code)

            if l < 64:
                accoeff = DecodeNumber(code, bits)
                acquant = accoeff * quant[l]
                Coeff[l] = acquant
                l += 1
        return Coeff, dccoeff


    def StartOfScan(self, data, hdrlen):
        data, lenchunk = RemoveFF00(data[hdrlen:])
        st = Stream(data)

        Lum_quant_table = self.quant[self.quantMapping[0]]
        Col_quant_table = self.quant[self.quantMapping[1]]

        oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
        y_blocks, Cr_blocks, Cb_blocks = [], [], []
        for y in range(self.height // 8):
            for x in range(self.width // 8):

                Y_block_coeff, oldlumdccoeff = self.BuildMatrix(st, 0, Lum_quant_table, oldlumdccoeff)
                Cr_block_coeff, oldCrdccoeff = self.BuildMatrix(st, 1, Col_quant_table, oldCrdccoeff)
                Cb_block_coeff, oldCbdccoeff = self.BuildMatrix(st, 1, Col_quant_table, oldCbdccoeff)

                y_DCT_block = np.zeros((8, 8), dtype='float64')
                Cr_DCT_block = np.zeros((8, 8), dtype='float64')
                Cb_DCT_block = np.zeros((8, 8), dtype='float64')
                for j in range(64):
                    row = zigzag[j] // 8
                    col = zigzag[j] % 8
                    y_DCT_block[row, col] = Y_block_coeff[j]
                    Cr_DCT_block[row, col] = Cr_block_coeff[j]
                    Cb_DCT_block[row, col] = Cb_block_coeff[j]

                y_block = cv2.idct(y_DCT_block)
                Cr_block = cv2.idct(Cr_DCT_block)
                Cb_block = cv2.idct(Cb_DCT_block)
                y_blocks.append(y_block)
                Cr_blocks.append(Cr_block)
                Cb_blocks.append(Cb_block)

        y_img = integrate_block(self.height, self.width, y_blocks)
        cr_img = integrate_block(self.height, self.width, Cr_blocks)
        cb_img = integrate_block(self.height, self.width, Cb_blocks)
        img = np.clip(cv2.merge([y_img, cb_img, cr_img])+128, 0, 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        return lenchunk + hdrlen, img


    def DefineQuantizationTables(self, data):
        while len(data) > 0:
            hdr, = unpack("B", data[0:1])
            self.quant[hdr & 0xf] = GetArray("B", data[1:1 + 64], 64)
            data = data[65:]

    def BaselineDCT(self, data):
        hdr, self.height, self.width, components = unpack(">BHHB", data[0:6])

        for i in range(components):
            id, samp, QtbId = unpack("BBB", data[6 + i * 3:9 + i * 3])
            self.quantMapping.append(QtbId)


    def DefineHuffmanTables(self, data):
        while len(data) > 0:
            off = 0
            hdr, = unpack("B", data[off:off + 1])
            off += 1

            lengths = GetArray("B", data[off:off + 16], 16)
            off += 16
            elements = []
            for i in lengths:
                elements += (GetArray("B", data[off:off + i], i))
                off = off + i

            hf = HuffmanTable()
            hf.GetHuffmanBits(lengths, elements)
            self.tables[hdr] = hf

            data = data[off:]


    def decode(self, path):
        data = open(path, 'rb').read()
        while True:
            hdr, = unpack(">H", data[0:2])
            if hdr == 0xffd8:
                lenchunk = 2
            elif hdr == 0xffd9:
                break
            else:
                lenchunk, = unpack(">H", data[2:4])
                lenchunk += 2
                chunk = data[4:lenchunk]
                if hdr == 0xffdb:
                    self.DefineQuantizationTables(chunk)
                elif hdr == 0xffc0:
                    self.BaselineDCT(chunk)
                elif hdr == 0xffc4:
                    self.DefineHuffmanTables(chunk)
                elif hdr == 0xffda:
                    lenchunk,img = self.StartOfScan(data, lenchunk)

            data = data[lenchunk:]
            if len(data) == 0:
                break
        return img