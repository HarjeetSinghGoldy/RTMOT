from queue import Empty
from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import random
import datetime

import multiprocessing as mp

from multiprocessing import Queue

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 1
nn_budget = None
nms_max_overlap = 1

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
frame_rate = 1


# vid = cv2.VideoCapture('./data/video/ped.mp4')


def reset_attempts():
    return 50


def process_video(attempts, camera):
    timeNow = str(time.strftime("%H%M%S"))
    fileName = './data/video/results' + timeNow + '.avi'
    t2 = str(random.randint(0, 9))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = int(camera.get(cv2.CAP_PROP_FPS))
    print("vid_fps",vid_fps)
    if vid_fps > 25:
        vid_fps = int(vid_fps)

    vid_width, vid_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(fileName, codec, vid_fps, (vid_width, vid_height))

    from _collections import deque

    pts = [deque(maxlen=30) for _ in range(1000)]

    counter = []
    person_counter = []

    prev = 0
    while True:
        time_elpased = time.time() - prev
        _, img = camera.read()
        # img = cam.get_frame(0.65)

        if not _:
            print("disconnected!")
            camera.release()
            out.release()
            print(fileName)

            cv2.destroyAllWindows()
            if attempts > 0:
                time.sleep(5)
                return True
            else:
                return False

        # if time_elpased > 1. / frame_rate:
        if True:

            prev = time.time()
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)
            t1 = time.time()

            boxes, scores, classes, nums = yolo.predict(img_in)

            classes = classes[0]
            names = []
            for i in range(len(classes)):
                names.append(class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(img, boxes[0])
            features = encoder(img, converted_boxes)

            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(converted_boxes, scores[0], names, features)]

            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            current_count = int(0)
            current_person_count = int(0)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]

                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)
                                                                                       + len(str(track.track_id))) * 17,
                                                                       int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)

                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                pts[track.track_id].append(center)

                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(img, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

                height, width, _ = img.shape
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

                # cv2.line(img, (int(width/2),0), (int(width/2), height),
                #          (255, 255, 0), thickness=2)

                # Vertical Area
                x = int(3 * width / 5 - width / 20) - 200
                gapPix = x + 200
                cv2.line(img, (x, 0), (x, height),
                         (0, 255, 0), thickness=4)
                cv2.line(img, (gapPix, 0), (gapPix, height),
                         (0, 255, 0), thickness=4)

                # Horizontal Area
                # cv2.line(img, (0, int(3 * height / 5 + height / 20)), (width, int(3 * height / 5 + height / 20)),
                #          (0, 255, 0), thickness=2)
                # cv2.line(img, (0, int(3 * height / 5 - height / 20)), (width, int(3 * height / 5 - height / 20)),
                #          (0, 255, 0), thickness=2)

                center_y = int(((bbox[1]) + (bbox[3])) / 2)
                center_x = int(((bbox[0]) + (bbox[2])) / 2)

                if str(track.track_id) == "18":
                    print(bbox[1])
                    # print(center_x, center_y, str(track.track_id))
                    # print(int(3 * width / 5 - width / 20))
                    # print(int(3 * width / 5 + width / 20))

                # if int(3 * width / 5 - width / 20) >= center_y >= int(3 * width / 5 + width / 20) :
                #     if class_name == 'car' or class_name == 'truck':
                #         counter.append(int(track.track_id))
                #         current_count += 1
                #     if class_name == 'person':
                #         person_counter.append(int(track.track_id))
                #         current_person_count += 1
                if gapPix >= center_x >= x:
                    if class_name == 'car' or class_name == 'truck':
                        counter.append(int(track.track_id))
                        current_count += 1
                    if class_name == 'person':
                        person_counter.append(int(track.track_id))
                        current_person_count += 1

            total_count = len(set(counter))
            total_person = len(set(person_counter))

            # cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
            # cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)
            cv2.putText(img, "Current Person Count: " + str(current_person_count), (0, 180), 0, 1, (0, 0, 255), 2)
            cv2.putText(img, "Total Person Count: " + str(total_person), (0, 230), 0, 1, (0, 0, 255), 2)

            fps = 1. / (time.time() - t1)
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
            cv2.namedWindow(fileName, 0);
            cv2.resizeWindow(fileName, 1024, 768)
            cv2.imshow(fileName, img)
            out.write(img)
            if cv2.waitKey(10) == ord('q'):
                print("forced complete")
                break


def process_video_queue(img,fileName,pts,counter,person_counter,prev,out):
    print("INnnnnnnnnnnnnnn")

    time_elpased = time.time() - prev
    if time_elpased > 1. / frame_rate:
        prev = time.time()
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)
            
        except Exception as e:
            print(e)
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)

        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                        zip(converted_boxes, scores[0], names, features)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        current_count = int(0)
        current_person_count = int(0)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)
                                                                                    + len(str(track.track_id))) * 17,
                                                                    int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            pts[track.track_id].append(center)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(img, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

            height, width, _ = img.shape
            #
            # cv2.line(img, (int(width/2),0), (int(width/2), height),
            #          (255, 255, 0), thickness=2)

            # Vertical Area
            x = int(3 * width / 5 - width / 20) - 200
            gapPix = x + 200
            cv2.line(img, (x, 0), (x, height),
                        (0, 255, 0), thickness=4)
            cv2.line(img, (gapPix, 0), (gapPix, height),
                        (0, 255, 0), thickness=4)

            # Horizontal Area
            # cv2.line(img, (0, int(3 * height / 5 + height / 20)), (width, int(3 * height / 5 + height / 20)),
            #          (0, 255, 0), thickness=2)
            # cv2.line(img, (0, int(3 * height / 5 - height / 20)), (width, int(3 * height / 5 - height / 20)),
            #          (0, 255, 0), thickness=2)

            center_y = int(((bbox[1]) + (bbox[3])) / 2)
            center_x = int(((bbox[0]) + (bbox[2])) / 2)

            if str(track.track_id) == "18":
                print()
                # print(center_x, center_y, str(track.track_id))
                # print(int(3 * width / 5 - width / 20))
                # print(int(3 * width / 5 + width / 20))

            # if int(3 * width / 5 - width / 20) >= center_y >= int(3 * width / 5 + width / 20) :
            #     if class_name == 'car' or class_name == 'truck':
            #         counter.append(int(track.track_id))
            #         current_count += 1
            #     if class_name == 'person':
            #         person_counter.append(int(track.track_id))
            #         current_person_count += 1
            if gapPix >= center_x >= x:
                if class_name == 'car' or class_name == 'truck':
                    counter.append(int(track.track_id))
                    current_count += 1
                if class_name == 'person':
                    person_counter.append(int(track.track_id))
                    current_person_count += 1

        total_count = len(set(counter))
        total_person = len(set(person_counter))

        # cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
        # cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Current Person Count: " + str(current_person_count), (0, 180), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Total Person Count: " + str(total_person), (0, 230), 0, 1, (0, 0, 255), 2)

        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
        cv2.namedWindow(fileName, 0)
        cv2.resizeWindow(fileName, 1024, 768)
        cv2.imshow(fileName, img)
        print(out)
        out.write(img)
        # if cv2.waitKey(10) == ord('q'):
        #     print("forced complete")




recall = True
attempts = reset_attempts()
while recall:
    camera = cv2.VideoCapture("rtsp://admin:root1234@192.168.1.64:554/out.h264")
    camera.set(3, 640)
    camera.set(3, 480)
    camera.set(cv2.CAP_PROP_FPS, 5)
    if camera.isOpened():
        print("[INFO] Camera connected at " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        attempts = reset_attempts()
        recall = process_video(attempts,camera)
    else:
        print("Camera not opened " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        camera.release()
        attempts -= 1
        print("attempts: " + str(attempts))

        # give the camera some time to recover
        time.sleep(5)
        continue

import os

from collections import deque
def writeQueue(queue,url,top) -> None:
    print('Process to write: %s' % os.getpid())
    camera = cv2.VideoCapture("rtsp://admin:root1234@192.168.1.64:554/out.h264")
    camera.set(3, 640)
    camera.set(3, 480)
    camera.set(cv2.CAP_PROP_FPS, 5)
    while True:
        _, img = camera.read()
        if not _:
            print("Not Frame")
            break
        queue.put(img)
        print(queue.qsize())
        if queue.qsize() >= top:
            queue.close()
            queue = mp.Queue()
            gc.collect()


def readQueue(queue) -> None:

    print('Process to read: %s' % os.getpid())
    timeNow = str(time.strftime("%H%M%S"))
    fileName = './data/video/results' + timeNow + '.avi'
    t2 = str(random.randint(0, 9))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = int(5)

    if vid_fps > 25:
        vid_fps = int(25)
    vid_fps = int(frame_rate)

    vid_width, vid_height = int(640), int(480)
    out = cv2.VideoWriter(fileName, codec, vid_fps, (vid_width, vid_height))

    from _collections import deque

    pts = [deque(maxlen=30) for _ in range(1000)]

    counter = []
    person_counter = []

    prev = 0
    while True:
        # print(len(queue))
        if not queue.empty():
            value = queue.get()
            # cv2.imshow("img", value)
            if len(value):
                process_video_queue(value,fileName,pts,counter,person_counter,prev,out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # camera.release()
                out.release()
                break

    


import os
import cv2
import gc
from multiprocessing import Process, Manager
import multiprocessing as mp




# # Write data to the shared buffer stack:
def write(stack, cam, top: int) -> None:
    """
         :param cam: camera parameters
         :param stack: Manager.list object
         :param top: buffer stack capacity
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    cap.set(3, 640)
    cap.set(3, 480)
    cap.set(cv2.CAP_PROP_FPS, 5)
    while True:
        _, img = cap.read()
        if not _:
            print("Not Frame")
            break
        stack.append(img)
        print(len(stack))
        # Clear the buffer stack every time it reaches a certain capacity
        # Use the gc library to manually clean up memory garbage to prevent memory overflow
        if len(stack) >= top:
            del stack[:]
            gc.collect()
     


# # Read data in the buffer stack:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    timeNow = str(time.strftime("%H%M%S"))
    fileName = './data/video/results' + timeNow + '.avi'
    t2 = str(random.randint(0, 9))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = int(25)

    if vid_fps > 25:
        vid_fps = int(25)
    vid_fps = int(frame_rate)

    vid_width, vid_height = int(640), int(480)
    out = cv2.VideoWriter(fileName, codec, vid_fps, (vid_width, vid_height))

    from _collections import deque

    pts = [deque(maxlen=30) for _ in range(1000)]

    counter = []
    person_counter = []

    prev = 0
    while True:
        if len(stack) != 0:
            value = stack.pop()
            # cv2.imshow("img", value)
            if len(value):
                process_video_queue(value,fileName,pts,counter,person_counter,prev,out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


# if __name__ == '__main__':
#     # The parent process creates a buffer stack and passes it to each child process:
#     mp.set_start_method('spawn')
#     q = mp.Queue()
#     pw = mp.Process(target=writeQueue, args=(q, "rtsp://admin:root1234@192.168.1.64:554/out.h264", 100))
#     pr = mp.Process(target=readQueue, args=(q,))
#     # Start the child process pw, write:
#     pw.start()
#     # Start the child process pr, read:
#     pr.start()


#     # Wait for pr to end:
#     pr.join()

#     # pw Process is an infinite loop, can not wait for its end, can only be forced to terminate:
#     pw.terminate()



# if __name__ == '__main__':
#     q = Manager().list()
#     pw = Process(target=write, args=(q, "rtsp://admin:root1234@192.168.1.64:554/out.h264", 500))
#     pr = Process(target=read, args=(q,))
#     # Start the child process pw, write:
#     pw.start()
#     # Start the child process pr, read:
#     pr.start()


#     # Wait for pr to end:
#     pr.join()

#     # pw Process is an infinite loop, can not wait for its end, can only be forced to terminate:
#     pw.terminate()
