from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets/examples', help='root directory for training data')
parser.add_argument('--output_folder', type=str, default='datasets/examples/detections', help='output folder')
opt = parser.parse_args()


def landmark_generate():
    # Create face detector
    mtcnn = MTCNN(keep_all=True, device='cuda:0')

    # Load images
    imgs_list = []
    imgs_list += [os.path.join(opt.data_root, i) for i in sorted(os.listdir(opt.data_root)) if 'jpg' in i or
                                                        'png' in i or 'jpeg' in i or 'PNG' in i]
    outputs_list = []
    outputs_list += [os.path.join(opt.output_folder, '/'.join(i.split('.')[:-1])) + '.txt' for i in
                     sorted(os.listdir(opt.data_root)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]

    print(imgs_list)
    print(outputs_list)
    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        output_path = outputs_list[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Detect face
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        # Visualize
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(img)
        ax.axis('off')
        np.set_printoptions(suppress=True)
        for box, landmark in zip(boxes, landmarks):
            ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
            print(output_path)
            print(landmark)
            np.savetxt(output_path, landmark, fmt='%.05f')
            break
        # fig.show()
        # plt.savefig('123.jpg')


if __name__ == '__main__':
    landmark_generate()