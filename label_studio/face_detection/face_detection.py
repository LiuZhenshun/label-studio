import cv2
import mediapipe as mp
import skvideo.io
import os
import shutil
import matplotlib.pyplot as plt
import sqlite3
import json

from label_studio.core.utils.io import get_data_dir

def face_detection_video(video_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"JPEG")
    #fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    video = cv2.VideoWriter('/media/hkuit164/Backup/tmp.mp4', fourcc, 30, (width,height))
    # Create a VideoCapture object and read from input file

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    i = 0
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        if ret == False:
            break
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                video.write(frame)
            else:
                annotated_image = frame
                for detection in results.detections:
                    #print('Nose tip:')
                    #print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    mp_drawing.draw_detection(annotated_image, detection)
                #plt.imshow(annotated_image)
                #plt.title('Matplotlib')
                #Give this plot a title,
                #so I know it's from matplotlib and not cv2
                #plt.show()
                video.write(annotated_image)
        i = i + 1
        ret, frame = cap.read()
        #print(ret)

    # When everything done, release
    # the video capture object
    cap.release()
    video.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def face_detection_image(path):
    mp_face_detection = mp.solutions.face_detection
    frame = cv2.imread(path)
    conn = sqlite3.connect(os.path.join(get_data_dir(),'label_studio.sqlite3'))
    print("Opened database successfully")
    cur = conn.cursor()
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            pass
        else:
            shape_img = frame.shape[0:2]
            value = []
            k = 0
            for i in results.detections:
                height = i.location_data.relative_bounding_box.height * 100
                width = i.location_data.relative_bounding_box.width * 100
                xmin = i.location_data.relative_bounding_box.xmin * 100
                ymin = i.location_data.relative_bounding_box.ymin * 100
                if k > 9:
                    id = 'selfdete0{}'.format(k)
                else:
                    id = 'selfdete{}'.format(k)
                value1 = {"original_width": shape_img[1], "original_height": shape_img[0], "image_rotation": 0, "value": {"x": xmin, "y": ymin,\
                            "width": width, "height": height, "rotation": 0, "rectanglelabels": ["Face"]},\
                            "id": id, "from_name": "label", "to_name": "image", "type": "rectanglelabels", "origin": "manual"}
                value.append(value1)
                k = k + 1
            serialised = json.dumps(value)
            max_id = cur.execute('SELECT MAX(id) FROM task_completion')
            temp = max_id.fetchone()[0]
            #print(temp)
            if temp==None:
               max_id_num = 1
            else:
               max_id_num = temp + 1

            temp1 = cur.execute('SELECT MAX(id) FROM task').fetchone()[0]
            if temp1==None:
               id_task = 1
            else:
               id_task = temp1 + 1

            cur.execute("INSERT INTO task_completion (id, result, was_cancelled, created_at,\
                        updated_at, task_id, prediction, lead_time, result_count, completed_by_id,\
                        ground_truth) VALUES \
                        (?,?,?,?,?,?,?,?,?,?,?)",(max_id_num ,serialised,0,'2022-08-18 02:00:40.147998','2022-08-23 07:01:27.825691',id_task,'{}',44.806,0,1,0))
            conn.commit()




#face_detection('/media/hkuit164/Backup/videoplayback1.mp4')
'''
    #shutil.move('/media/hkuit164/Backup/videoplayback2.mp4', video_path)
face_detection('/media/hkuit164/Backup/videoplayback1.mp4')

kk = skvideo.io.vread('/media/hkuit164/Backup/videoplayback1.mp4')
outputfile = "/media/hkuit164/Backup/videoplayback4.mp4"

#writer = skvideo.io.FFmpegWriter(outputfile, outputdict={'-vcodec': 'h264'})
writer = skvideo.io.FFmpegWriter(outputfile)

for frame in kk:
    writer.writeFrame(frame)
writer.close()
'''
#face_detection_image('/media/hkuit164/Backup/test_face.jpg',106)
