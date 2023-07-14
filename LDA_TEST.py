import matplotlib
import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
from HadirApp.models import Student
import shutil


from RecognitionSystem.FaceDetection import resource_path
img_classes = 0
width, height = (112, 92)
img_for_each_class = 5
test_path = resource_path('RecognitionSystem/Processing/Detections')
img_format = 'pgm'
matplotlib.use('tkagg')


def DeleteFolderIfExist(Path):
    for filename in os.listdir(Path):
        file_path = os.path.join(Path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def img_to_vector(filename):
    img = Image.open(filename).convert('L').resize((height, width))
    img_vector = np.array(img).flatten()
    sz = img_vector.shape[0]
    return img_vector


def find_mean(arr):
    M = np.mean(arr, axis=1)
    return M


def mean_normalization(arr):
    M = find_mean(arr)
    x_range = arr.shape[0]
    y_range = arr.shape[1]
    for i in range(x_range):
        for j in range(y_range):
            arr[i][j] -= M[i]
    return arr


def Create_Temp_Dataset(students=[]):
    import os
    global img_for_each_class

    student_classes_info = []
    dataset_path = resource_path(
        'RecognitionSystem/Processing/Recognition/Dataset')
    saved_images_path = resource_path('HadirApp/media/Students')

    img_for_each_class = 900
    for folder in os.listdir(saved_images_path):
        if folder in students:
            size = len(os.listdir(f'{saved_images_path}/{folder}'))
            if size < img_for_each_class:
                img_for_each_class = size

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    current_class_number = 1
    for folder in os.listdir(saved_images_path):
        images_folder = resource_path(f'HadirApp/media/Students/{folder}')
        num_of_images = len(os.listdir(images_folder))
        loop_counter = int(num_of_images / img_for_each_class)
        current_loop = 0

        # How many discarded images ?
        taken_images = num_of_images - (num_of_images % img_for_each_class)
        current_image_count = 1  # this is a reset to take new images
        class_idx_start = current_class_number

        for file in os.listdir(images_folder):  # loop through images
            if current_loop < loop_counter:

                if current_image_count > taken_images:
                    break

                images_file = resource_path(
                    f'HadirApp/media/Students/{folder}/{file}')
                dataset_path_counter = resource_path(
                    f'RecognitionSystem/Processing/Recognition/Dataset/{current_class_number}')

                if not os.path.exists(dataset_path_counter):
                    os.makedirs(dataset_path_counter)

                shutil.copy(images_file, dataset_path_counter)

                if current_image_count % img_for_each_class == 0:
                    current_class_number += 1
                    current_loop += 1

                current_image_count += 1

        class_idx_stop = current_class_number - 1
        student_classes_info.append(
            [str(folder), str(class_idx_start), str(class_idx_stop)])
    return np.array(student_classes_info)


def preprocess2():
    train_set = []
    train_set_number = []
    test_set = []
    test_set_number = []

    dataset_dir = resource_path(
        'RecognitionSystem/Processing/Recognition/Dataset')
    test_dir = resource_path('RecognitionSystem/Processing/Detections')

    for folder in os.listdir(dataset_dir):
        for file in os.listdir(f'{dataset_dir}/{folder}'):
            if file.endswith('.pgm'):
                path = f'{dataset_dir}/{folder}/{file}'
                img = img_to_vector(path).astype(np.int64)
                train_set.append(img)
                train_set_number.append(folder)

    i = 1
    for file in os.listdir(f'{test_dir}'):
        if file.endswith('.pgm'):
            path = f'{test_dir}/{file}'
            img = img_to_vector(path).astype(np.int64)
            test_set.append(img)
            test_set_number.append(i)
            i += 1

    train_set = np.array(train_set)
    train_set_number = np.array(train_set_number)
    test_set = np.array(test_set)
    test_set_number = np.array(test_set_number)

    global img_classes
    img_classes = len(os.listdir(dataset_dir))

    return train_set.T, train_set_number, test_set.T, test_set_number


def Recognize_LDA(students):
    debugMode = False
    returned_info = []

    ids = []
    for std in students:
        ids.append(str(std.student_id))

    print(f'We have {len(ids)} students in this class : {ids}')

    student_classes_info = Create_Temp_Dataset(ids)
    train_set, train_set_number, test_set, test_set_number = preprocess2()
    if debugMode:
        for i in range(0, 8*img_for_each_class):
            plt.subplot(8, img_for_each_class, i + 1)
            plt.imshow(train_set[:, i].reshape(width, height), cmap='gray')
            plt.title(f'Training { i }')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)
        plt.show()

    # Mean
    M = find_mean(train_set)
    if debugMode:
        plt.subplot(1, 1, 1)
        plt.imshow(M[:,].reshape(width, height), cmap='gray')
        plt.title(f'Mean')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                        top='off', right='off', left='off', which='both')
        plt.subplots_adjust(hspace=0.9)
        plt.show()

    # Norm mean
    A = mean_normalization(train_set)
    print(A.shape)
    if debugMode:
        for i in range(0, 8*img_for_each_class):
            plt.subplot(8, img_for_each_class, i + 1)
            plt.imshow(A[:, i].reshape(width, height), cmap='gray')
            plt.title("Norm mean")
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)
    if debugMode:
        plt.show()

    L = np.dot(A.T, A)
    print(f'L Shape {L.shape}')

    eigenvalues, eigenvectors = np.linalg.eig(L)
    # sort eigenvectors by the value of eigenvalues
    print("eigenvalues shape : ", eigenvalues.shape)
    print("eigenvectors shape : ", eigenvectors.shape)

    U = []
    sz = L.shape[0]
    for i in range(sz):
        U.append(np.dot(A, eigenvectors[:, i]))

    U = np.array(U)
    U = U.T

    print("U shape : ", U.shape)
    if debugMode:
        for i in range(0, 8*img_for_each_class):
            plt.subplot(8, img_for_each_class, i+1)
            plt.imshow(U[:, i].reshape(width, height), cmap='gray')
            plt.title("U")
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)
    if debugMode:
        plt.show()

    weight_vector = np.dot(U.T, A)
    print("weight_vector shape : ", weight_vector.shape)
    overall_mean = np.mean(weight_vector, axis=1)
    overall_mean = overall_mean.reshape(overall_mean.shape[0], 1)

    SW = np.zeros([train_set.shape[1], train_set.shape[1]])
    for i in range(img_classes):
        ind = i * img_for_each_class
        V = weight_vector[:, ind:ind+img_for_each_class]
        mean_local = np.mean(V, axis=1)
        mean_local = mean_local.reshape(mean_local.shape[0], 1)
        mean = np.repeat(mean_local, img_for_each_class, axis=1)
        diff = V - mean
        variance = np.dot(diff, diff.T)
        SW = SW + variance
    print("SW shape : ", SW.shape)

    # Finding the between class scatter matrix
    SB = np.zeros([train_set.shape[1], train_set.shape[1]])
    for i in range(img_classes):
        j = i + img_for_each_class
        V = weight_vector[:, i:j]
        mean_local = np.mean(V, axis=1)
        mean_local = mean_local.reshape(mean_local.shape[0], 1)
        diff = mean_local - overall_mean
        sigma = np.dot(diff, mean_local.T)
        SB = SB + sigma
    print("SB shape : ", SW.shape)

    # finding the criterion function
    # this function maximises the between class scatter and minimizes the within class scatter
    J = np.dot(np.linalg.pinv(SW), SB)
    print("J shape : ", J.shape)

    # finding the criterion function
    # this function maximises the between class scatter and minimizes the within class scatter
    eigenval, eigenvec = np.linalg.eig(J)
    fisher_faces = np.dot(eigenvec.T, weight_vector)
    print("Fisher face shape", fisher_faces.shape)

    # Testing

    debugMode = False

    x_range = test_set.shape[0]
    y_range = test_set.shape[1]
    if debugMode:
        plt.subplot(1, 1, 1)
        plt.imshow(test_set[:,].reshape(width, height), cmap='gray')
        plt.title(f'Test image')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                        top='off', right='off', left='off', which='both')
        plt.subplots_adjust(hspace=0.9)
    plt.show()

    for i in range(x_range):
        for j in range(y_range):
            test_set[i][j] -= M[i]
    debugMode = False
    if debugMode:
        for i in range(0, test_set.shape[1]):
            plt.subplot(1, test_set.shape[1], i + 1)
            plt.imshow(test_set[:, i].reshape(width, height), cmap='gray')
            plt.title("Test")
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)
    if debugMode:
        plt.show()

    weight_vector_test = np.dot(U.T, test_set)
    projected_fisher_faces = np.dot(eigenvec.T, weight_vector_test)

    skip_this_person = False
    for i in range(test_set.shape[1]):
        ith_wv = projected_fisher_faces[:, i]
        ans = 0
        index = 0

        for j in range(train_set.shape[1]):
            jth_wv = fisher_faces[:, j]
            diff = ith_wv - jth_wv
            diff = np.absolute(diff)
            sm = np.sum(diff)

            if ans == 0:
                ans = sm
                index = j
            else:
                if sm < ans:
                    ans = sm
                    index = j

        if debugMode:
            plt.subplot(1, 2, 1)
            plt.imshow(test_set[:, i].reshape(width, height), cmap='gray')
            plt.title(f'Test: ')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)

            plt.subplot(1, 2, 2)
            plt.imshow(train_set[:, index].reshape(width, height), cmap='gray')
            txt = 'Matched: '
            matched = train_set_number[index]
            for item in student_classes_info:
                if int(item[1]) <= int(matched) <= int(item[2]):
                    txt = str(item[0])
            plt.title(txt)
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',
                            top='off', right='off', left='off', which='both')
            plt.subplots_adjust(hspace=0.9)
            plt.show()

        matched = train_set_number[index]
        for item in student_classes_info:
            if int(item[1]) <= int(matched) <= int(item[2]):
                returned_info.append(str(item[0]))

    return returned_info
