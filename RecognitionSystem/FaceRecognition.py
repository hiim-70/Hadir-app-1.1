# Everything is explained here
# https://github.com/vutsalsinghal/EigenFace/blob/master/Face%20Recognition.ipynb

from matplotlib import pyplot as plt
from matplotlib.image import imread
import matplotlib
import numpy as np
import os

from HadirApp.models import *
from RecognitionSystem.FaceDetection import resource_path

count = 0
num_images = 0
correct_pred = 0
width, height = 195, 231
debugMode = False


def Recognize_TestImages(Result, img, training_tensor, training_IDs, proj_data, mean_face, w):
    global count, highest_min, num_images, correct_pred, width, height, debugMode
    unknown_face = img
    num_images += 1

    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    # Weight
    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    # Thresholds 1,2
    t0 = 600535910  # t0 = 200535910  True
    t1 = 358850297.2102374  # t1 = 143559033    False

    # Store results here for future uses
    if training_IDs[index] not in Result:

        if norms[index] < t0 and norms[index] > t1:
            Result.append(training_IDs[index])
        else:
            print(f" No match for {training_IDs[index]} (index out of range)")
    print(norms[index], training_IDs[index])
    if debugMode:
        if norms[index]:
            plt.subplot(2, 2, 1+count)
            if norms[index] < t0:  # It's a face
                plt.title('Matched:'+f' {training_IDs[index]}', color='g')
                plt.imshow(training_tensor[index, :].reshape(
                    height, width), cmap='gray')
            else:
                print(" Not a human.")
        else:
            plt.subplot(2, 2, 1+count)
            if norms[index] < t0:  # It's a face
                plt.title('Similar to:'+f' {training_IDs[index]}', color='b')
                plt.imshow(training_tensor[index, :].reshape(
                    height, width), cmap='gray')
            else:
                plt.title('almost looks like:' +
                          f' {training_IDs[index]}', color='b')
                plt.imshow(training_tensor[index, :].reshape(
                    height, width), cmap='gray')


def Recognize(students):
    global width, height, debugMode

    TestPath = resource_path('RecognitionSystem/Processing/Detections')
    matplotlib.use('tkagg')

    wantedImgs = []
    for s in students:
        for img in Image.objects.filter(student=s):
            wantedImgs.append(img)

    for img in Traning.objects.all():
        wantedImgs.append(img)

    ImageSet = wantedImgs
    ImageSet_Size = len(ImageSet)
    TestImage_Size = len(os.listdir(TestPath))

    # Store students Images
    training_tensor = np.ndarray(
        shape=(ImageSet_Size, width*height), dtype=np.float64)
    training_IDs = []

    idx = 0
    for imgData in ImageSet:
        img = plt.imread(imgData.images)
        training_tensor[idx, :] = np.array(img, dtype=np.float64).flatten()
        training_IDs.append(imgData.student.student_id)
        if idx <= 23 and debugMode:
            plt.subplot(5, 5, 1+idx)
            plt.imshow(img, cmap='gray')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
        idx += 1

    # Store Test Images
    testing_tensor = np.ndarray(
        shape=(len(os.listdir(TestPath)), width*height), dtype=np.float64)
    idx = 0
    for file in os.listdir(TestPath):
        if file.endswith('.jpg'):
            full_path = TestPath + '/' + file
            img = plt.imread(full_path)
            testing_tensor[idx, :] = np.array(img, dtype=np.float64).flatten()
            idx += 1

    ################## Calculation ##################

    # Mean face
    mean_face = np.zeros((1, height * width))

    for i in training_tensor:
        mean_face = np.add(mean_face, i)

    mean_face = np.divide(mean_face, float(ImageSet_Size)).flatten()

    # Normalised faces
    normalised_training_tensor = np.ndarray(
        shape=(ImageSet_Size, height * width))
    for i in range(ImageSet_Size):
        normalised_training_tensor[i] = np.subtract(
            training_tensor[i], mean_face)

    for i in range(ImageSet_Size):
        img = normalised_training_tensor[i].reshape(height, width)

    # Normalised faces
    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix, 8.0)

    # Normalised faces
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index])
                 for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1]
                       for index in range(len(eigenvalues))]

    # PCA K -> 7
    reduced_data = np.array(eigvectors_sort[:20]).transpose()
    proj_data = np.dot(training_tensor.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    # Plot EigenFaces
    for i in range(proj_data.shape[0]):
        img = proj_data[i].reshape(height, width)
        if idx <= 23 and debugMode:
            plt.subplot(1, proj_data.shape[0], 1+i)
            plt.imshow(img, cmap='gray')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
    if debugMode:
        plt.show()

    # Finding weights for each training image
    w = np.array([np.dot(proj_data, i) for i in normalised_training_tensor])

    # Recognition Place == Matching stored data against Taken image
    Recognition_Results = []
    debugMode = False
    for i in range(TestImage_Size):
        Recognize_TestImages(
            Recognition_Results, testing_tensor[i, :], training_tensor, training_IDs, proj_data, mean_face, w)
        if debugMode:
            plt.show()
    return Recognition_Results
