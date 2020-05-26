# https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm

import re
import numpy
import numpy as np
import os as os

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def convert_sif_pgm(filename):
    """ read sif file and save to pgm """

    x=0
    y=0
    num=0
    a=[]
    rows=2048 # set according to your detector
    data=[]
    row=[]
    file=""
    os.system("hexdump "+filename+">"+filename+".tmp")
    with open(filename+".tmp") as f:
        for line in f:
            line = line.split(' ')
            if len(line) > 1:
                rawdata.extend(line[1:])
    os.system("rm "+filename+".tmp") #print file

    pos=0
    while rawdata[pos]!="0044":
        pos=pos+1
    pos=pos-1
    y=0

    while pos<len(rawdata)-1:
        px=int(rawdata[pos],16)
        bit=rawdata[pos+1]
        if bit=="0045":
            px=px+2**16

        row.append(px)
        if len(row)==rows:
            data.append(row)
            row=[]
        pos=pos+2

    f = open(filename.replace(".sif", ".pgm"), "w")

    f.write("P2\n")
    f.write(str(len(data[0]))+" "+str(len(data))+" "+str(2**16-1)+"\n")

    for i in data:
        line=""
        for j in i:
            line=line+str(j/2)+" "
        f.write(line+"\n")
    f.close()


if __name__ == "__main__":
    from matplotlib import pyplot
    image = read_pgm("foo.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()
