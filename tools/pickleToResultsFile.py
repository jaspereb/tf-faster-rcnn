#Load the pickle results file produced by test.py and turn it into a VOC format results.txt file

#In the pickle file, the first dimension is the class (0-background, 1-plum), and the 2nd dim is the image number
#Each detection is (x1, y1, x2, y2, score)

import os
import pickle

dirPath = '/media/jasper/bigData/FasterRCNN-Output/Results/NightTransferResults'
filePath = os.path.join(dirPath, 'detections.pkl')
imPath = os.path.join(dirPath, 'detectedImgs.pkl')
outFile = os.path.join(dirPath, 'detections.txt')

print("Loading results file {}".format(filePath))
print("And saving results file {}".format(outFile))

with open(filePath, 'rb') as f:
    data = pickle.load(f)
    image_data = data[1] #Get the plum class

with open(imPath, 'r') as f:
    images = pickle.load(f)

assert len(image_data) == len(images) #Should be one image per detected frame

with open(outFile, 'w') as f:
    for i in range(0, len(image_data)):
        image_i = image_data[i]

        for n in range(0, len(image_i)):
            data_i = image_i[n]

            score = data_i[4]

            xmin = int(data_i[0])
            ymin = int(data_i[1])
            xmax = int(data_i[2])
            ymax = int(data_i[3])

            if(ymax > 480 or xmax > 640 or ymin < 0 or xmin < 0):
                print("Got detection outside image!!!")
                continue

            if(score < 0.01):
                continue

            imgName = images[i]
            dataLine = imgName + ' ' + str(score) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n'
            # print(dataLine)

            f.write(dataLine)

print("Done~!")
