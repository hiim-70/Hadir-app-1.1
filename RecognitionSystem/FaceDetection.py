
import os
import time
import keyboard

import numpy as np
import glob
import shutil
import mediapipe as mp

import skimage.exposure

# For yolo
import torch
from RecognitionSystem.models.common import DetectMultiBackend
from RecognitionSystem.utils.datasets import LoadImages, LoadStreams
from RecognitionSystem.utils.general import (
    check_img_size, cv2, increment_path, non_max_suppression, scale_coords, Profile)
from RecognitionSystem.utils.plots import Annotator, colors, save_one_box
from RecognitionSystem.utils.torch_utils import select_device, time_sync


def DeleteFolder(Path):
    for filename in os.listdir(Path):
        file_path = os.path.join(Path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = os.path.abspath(".")
    Parent = os.path.dirname(os.path.join(base_path, relative_path))
    if not os.path.exists(Parent):
        os.makedirs(os.path.dirname(os.path.join(
            base_path, relative_path).replace('\\', '/')))
    return os.path.join(base_path, relative_path).replace('\\', '/')


def DetectFaces(ImgSize=(256, 256), ImagePath='', Webcam=False, ConfidenceThreshold=.8, MaxDetection=100, Save=True, MediaPath=resource_path('HadirApp/media/Students'), std_id=-1, Save_cropped=True, Save_noBG=False, pad=10, gain=1.01):
    # Model Settings
    device = select_device('cpu')
    FaceModel = DetectMultiBackend(resource_path('RecognitionSystem/weights/best_FaceDetection.pt'),
                                   device=device, dnn=False, data=resource_path('RecognitionSystem/data.yaml'), fp16=False)
    stride = FaceModel.stride
    names = FaceModel.names
    pt = FaceModel.pt

    DetectedFilesName = []

    height, width = 92, 112

    AnnotationPath = ImagePath + 'Annotations'
    DetectedPath = ImagePath + 'Detections'
    NoBGPath = ImagePath + 'NoBG'

    save_path = f'{MediaPath}/{std_id}'

    images_count = 0

    # # Create folders to save images
    if os.path.exists(AnnotationPath) is False:
        os.mkdir(AnnotationPath)
    if os.path.exists(DetectedPath) is False:
        os.mkdir(DetectedPath)
    if os.path.exists(NoBGPath) is False:
        os.mkdir(NoBGPath)
    if os.path.exists(save_path) is False and std_id != -1:
        os.makedirs(save_path)

    # I manually added them here, they are function parameters by default
    global dir, num_tests
    visualize = False
    augment = False
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1)

    if Webcam:
        dataset = LoadStreams(0, img_size=ImgSize, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(ImagePath, img_size=ImgSize,
                             stride=stride, auto=pt)
        bs = 1  # batch_size

    FaceModel.warmup(imgsz=(1 if pt else bs, 3, *ImgSize))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # for every image
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if FaceModel.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = FaceModel(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, ConfidenceThreshold, 0.45, None, False, max_det=MaxDetection)

        # Start Prediction
        for i, det in enumerate(pred):  # Every detection on every image
            seen += 1

            if Webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            DetectedImage = im0.copy()  # To save image
            annotator = Annotator(im0, line_width=2, example=str(names))

            # Number of detection
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    images_count += 1

                    c = int(cls)  # integer class
                    annotator.box_label(
                        xyxy, f'{names[c]} {conf:.2f}', color=colors(c, True))

                    # 1.3 good for take images to dataset
                    # this was the best
                    # CroppedImg = save_one_box(xyxy, DetectedImage,gain=1.12, pad=60, BGR=True, save=False)
                    CroppedImg = save_one_box(
                        xyxy, DetectedImage, gain=gain, pad=pad, square=False, BGR=True, save=False)
                    CroppedImg = cv2.resize(CroppedImg, (height, width))
                    RGB = cv2.cvtColor(CroppedImg, cv2.COLOR_BGR2RGB)

                    # get the result
                    results = selfie_segmentation.process(RGB)
                    mask = results.segmentation_mask
                    condition = np.stack(
                        (results.segmentation_mask,) * 3, axis=-1) > 0.5
                    img_1 = np.zeros([165, 191, 3], dtype=np.uint8)
                    img_1.fill(255)
                    bg_image = cv2.resize(img_1, (height, width))

                    output_image = np.where(condition, CroppedImg, bg_image)

                    T_Cropped = cv2.cvtColor(CroppedImg, cv2.COLOR_BGR2GRAY)
                    T_NoBG = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

                    img = T_NoBG.copy()
                    blur = cv2.GaussianBlur(
                        img, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
                    result = skimage.exposure.rescale_intensity(
                        blur, in_range=(0, 255), out_range=(0, 255))

                    if Save:
                        if Save_cropped:
                            num_saved_img = len(os.listdir(save_path))
                            cropped_img_name = f'{save_path}/{int(num_saved_img)}'
                            cv2.imwrite(f'{cropped_img_name}.pgm', T_Cropped)
                            DetectedFilesName.append(f'{cropped_img_name}.pgm')

                        if Save_noBG:
                            num_saved_img = len(os.listdir(save_path))
                            cropped_img_name = f'{save_path}/{int(num_saved_img)}'
                            cv2.imwrite(f'{cropped_img_name}_noBG.pgm', result)
                            DetectedFilesName.append(
                                f'{cropped_img_name}_noBG.pgm')

                    if Save_cropped:
                        num_saved_img = len(os.listdir(DetectedPath))
                        cropped_img_name = f'{DetectedPath}/{int(num_saved_img)}'
                        cv2.imwrite(f'{cropped_img_name}.pgm', T_Cropped)

                    if Save_noBG:
                        num_saved_img = len(os.listdir(DetectedPath))
                        cropped_img_name = f'{DetectedPath}/{int(num_saved_img)}'
                        cv2.imwrite(f'{cropped_img_name}_noBG.pgm', result)

                cv2.imwrite(
                    f'{AnnotationPath}/{len(os.listdir(AnnotationPath))}.jpg', annotator.result())
                cv2.destroyAllWindows()

    return DetectedFilesName
