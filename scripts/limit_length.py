import sys

if __name__ == '__main__':

    filename = sys.argv[1]
    length = int(sys.argv[2])
    delim = " "

    fin = open(filename, 'r')
    fout = open("%s-%d" % (filename, length), 'w')

    for line in fin:
        if (len(line.split(delim)) > length):
            continue
        fout.write(line)

    fin.close()
    fout.close()

    pass
