import os
import re

new_txt = open('train_labels', 'w')
for dirname, dirnames, filenames in os.walk('./Main'):
    for filename in filenames:
        tmp = re.split('_', str(filename).strip())
        if tmp[1] == 'train.txt':
            old_txt = open(filename, 'r')
        
            for line in old_txt:
                temp = re.split(' ', line.strip())
                if temp[0] == '1':
                    new_txt.write(line + '\n')
        