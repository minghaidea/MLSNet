import csv


def ShapeRToCsv(path,path2, shapes, seq_len):
    for shape in shapes:
        i_file = open(path + '.' + shape)
        p='../Datasets/data/Shape/'
        o_file = csv.writer(open(p+'Train_' + shape + '.csv', 'w', newline=''))

        row = []
        for i in range(seq_len):
            row.append(i+1)

        for line in i_file.readlines():
            line = line.replace('\n', '')
            if line[0] == '>':
                o_file.writerow(row)
                row = []
            else:
                line = line.split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
        o_file.writerow(row)


    for shape in shapes:
        i_file = open(path2 + '.' + shape)
        p='../Datasets/data/Shape/'
        o_file = csv.writer(open(p+'Test_' + shape + '.csv', 'w', newline=''))

        row = []
        for i in range(seq_len):
            row.append(i+1)

        for line in i_file.readlines():
            line = line.replace('\n', '')
            if line[0] == '>':
                o_file.writerow(row)
                row = []
            else:
                line = line.split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
        o_file.writerow(row)


ShapeRToCsv(path='../Datasets/cache/train.txt',
            path2='../Datasets/cache/test.txt',
            shapes=['EP', 'HelT', 'MGW', 'ProT', 'Roll'], seq_len=101)

print('Success in getting shapes!')

