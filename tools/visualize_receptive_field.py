from tkinter import *
import numpy as np
from PIL import Image
from PIL import ImageTk
import cv2
import math
import pickle

global input_img
parent_units = {}
units_count = {}

class Feature_map:
    def __init__(self, layer_num, height, width):
        print('create feature map')
        self.layer_num = layer_num
        feature_map = np.empty([height, width], dtype=np.uint32)
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                idx = int('%04d' % self.layer_num + '%04d' % i + '%04d' % j)
                feature_map[i][j] = idx
                parent_units[str(idx)] = []
                units_count[str(idx)] = 0
        self.feature_map = feature_map
        self.height = height
        self.width = width
        self.count = np.zeros([self.height, self.width])

def conv(input, kernel_height=3, kernel_width=3, stride=1, pad=1):
    output_height = math.floor((input.height + 2 * pad - kernel_height) / stride) + 1
    output_width = math.floor((input.width + 2 * pad - kernel_width) / stride) + 1
    output_layer_num = input.layer_num + 1
    output = Feature_map(output_layer_num, output_height, output_width)
    for i in range(output.height):
        for j in range(output.width):
            # for every unit in output feature map
            parent_units[str(output.feature_map[i][j])] = np.array([], dtype=np.uint32)
            units_count[str(output.feature_map[i][j])] = 0
            name = str(output.feature_map[i][j])
            x_start = max(0, i*stride-pad)
            x_end = min(i*stride-pad + kernel_height, input.height)
            y_start = max(0, j*stride-pad)
            y_end = min(j*stride-pad + kernel_width, input.width)
            parent_units[str(output.feature_map[i][j])] = np.append(parent_units[str(output.feature_map[i][j])], input.feature_map[x_start:x_end, y_start:y_end])
    return output

def pooling(input, kernel_height=2, kernel_width=2, stride_x=2, stride_y=2):
    output_height = math.floor((input.height - kernel_height) / stride_x) + 1
    output_width =  math.floor((input.width - kernel_width) / stride_y) + 1
    output_layer_num = input.layer_num + 1
    output = Feature_map(output_layer_num, output_height, output_width)
    for i in range(output.height):
        for j in range(output.width):
        # for every unit in output feature map
            parent_units[str(output.feature_map[i][j])] = np.array([], dtype=np.uint32)
            units_count[str(output.feature_map[i][j])] = 0
            name = str(output.feature_map[i][j])
            x_start = max(0, i * stride_x)
            x_end = min(i * stride_x + kernel_height, input.height)
            y_start = max(0, j * stride_y)
            y_end = min(j * stride_y + kernel_width, input.width)
            parent_units[str(output.feature_map[i][j])] = np.append(parent_units[str(output.feature_map[i][j])], input.feature_map[x_start:x_end, y_start:y_end])
    return output

def create_nets():
    global input_img
    input_img = Feature_map(0, img_height, img_width)
    feature_map_1 = conv(input_img)
    feature_map_2 = conv(feature_map_1)
    feature_map_3 = pooling(feature_map_2)  # 15
    feature_map_4 = conv(feature_map_3)
    feature_map_5 = conv(feature_map_4)
    feature_map_6 = pooling(feature_map_5)  # 8
    feature_map_7 = conv(feature_map_6)
    feature_map_8 = conv(feature_map_7)
    feature_map_9 = conv(feature_map_8)
    feature_map_10 = pooling(feature_map_9)  # 4
    feature_map_11 = conv(feature_map_10)
    feature_map_12 = conv(feature_map_11)
    feature_map_13 = conv(feature_map_12)  # 4
    feature_map_14 = pooling(feature_map_13)
    # feature_map_14 = pooling(feature_map_13, kernel_width=1, stride_y=1)
    feature_map_15 = conv(feature_map_14)
    feature_map_16 = conv(feature_map_15)
    feature_map_17 = conv(feature_map_16)
    # file1 = open('input_img.pkl', 'wb')
    # pickle.dump(input_img, file1)
    # file1.close()
    # file2 = open('parent_units.pkl', 'wb')
    # pickle.dump(parent_units, file2)
    # file2.close()
    # file3 = open('units_count.pkl', 'wb')
    # pickle.dump(units_count, file3)
    # file3.close()
    return feature_map_17.layer_num, feature_map_17.height, feature_map_17.width

def load_net():
    global input_img, parent_units, units_count
    file1 = open('input_img.pkl', 'rb')
    input_img = pickle.load(file1)
    file1.close()
    file2 = open('parent_units.pkl', 'rb')
    parent_units = pickle.load(file2)
    file2.close()
    file3 = open('units_count.pkl', 'rb')
    units_count = pickle.load(file3)
    file3.close()

def update(name):
    global input_img
    units_count_copy = units_count.copy()
    query_identifier = name
    print(query_identifier)
    queue = []
    for parent in parent_units[query_identifier]:
        parent = str(parent)
        if parent == 'dummy':
            continue
        units_count_copy[parent] = units_count_copy[parent] + 1
        queue.append(parent)
    for unit in queue:
        for parent in parent_units[unit]:
            parent = str(parent)
            if parent == 'dummy':
                continue
            if units_count_copy[parent] == 0:
                queue.append(parent)
            units_count_copy[parent] = units_count_copy[parent] + units_count_copy[unit]
    input_count = np.zeros([img_height, img_width], dtype=np.int)
    for x in range(input_count.shape[0]):
        for y in range(input_count.shape[1]):
            input_count[x][y] = units_count_copy[str(input_img.feature_map[x][y])]
    return input_count

def callback(event):
    feature_y = event.x // rectangle_size
    feature_x = event.y // rectangle_size
    print("clicked at", feature_x, feature_y)
    name = '%d' % feature_num + '%04d' % feature_x + '%04d' % feature_y
    image = update(name)
    image = image.astype(np.float)
    image = image / np.max(image)
    image = image * 255
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panelA.configure(image=image)
    panelA.image = image

if __name__ == '__main__':
    rectangle_size = 20
    load_existing_model = False
    if load_existing_model:
        img_height = 600
        img_width = 800
        feature_num, feature_height, feature_width = 17, 38, 50
        load_net()
    else:
        img_height = 120
        img_width = 160
        feature_num, feature_height, feature_width = create_nets()

    root = Tk()
    image = np.zeros([img_height, img_width])
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panelA = Label(root, image=image)
    panelA.image = image
    panelA.pack(side="left")
    w = Canvas(root, width=rectangle_size * feature_width, height=rectangle_size * feature_height)
    w.pack(side="right")
    w.bind("<Button-1>", callback)
    for j in range(feature_height):
        for i in range(feature_width):
            w.create_rectangle(i * rectangle_size, j * rectangle_size, i * rectangle_size + rectangle_size,
                               j * rectangle_size + rectangle_size, fill="#476042")

    root.mainloop()