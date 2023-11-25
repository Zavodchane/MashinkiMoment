import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import os

'''
!!! Перед запуском скрипта создать папки jsons, result и vid !!!
'''

class Transport:
    def __init__(self, idx: str) -> None:
        self.first_line: bool = False
        self.second_line: bool = False
        self.inside_area: bool = False

        self.idx: str = idx
        self.ts_type: list = []

        self.frames: list = []


def get_crop_x(annotated_frame, data):
    areas = data["areas"][0]
    x_min = 10000
    x_max = 0
    for point in areas:
        x, y = int(point[0] * annotated_frame.shape[1]), int(point[1] * annotated_frame.shape[0])
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x

    return x_min, x_max


def is_inside_zone(point_x, point_y, frame) -> bool:
    areas = []
    areas1 = data['areas'][0]

    p1 = int(areas1[0][0] * frame.shape[1]), int(areas1[0][1] * frame.shape[0])
    p2 = int(areas1[1][0] * frame.shape[1]), int(areas1[1][1] * frame.shape[0])
    p3 = int(areas1[2][0] * frame.shape[1]), int(areas1[2][1] * frame.shape[0])
    p4 = int(areas1[3][0] * frame.shape[1]), int(areas1[3][1] * frame.shape[0])

    areas.append(np.array([p1, p2, p3, p4]))

    try:
        areas2 = data['areas'][1]

        p1 = int(areas2[0][0] * frame.shape[1]), int(areas2[0][1] * frame.shape[0])
        p2 = int(areas2[1][0] * frame.shape[1]), int(areas2[1][1] * frame.shape[0])
        p3 = int(areas2[2][0] * frame.shape[1]), int(areas2[2][1] * frame.shape[0])
        p4 = int(areas2[3][0] * frame.shape[1]), int(areas2[3][1] * frame.shape[0])
        areas.append(np.array([p1, p2, p3, p4]))
    except:
        pass

    for area in areas:
        polyTestRes = cv2.pointPolygonTest(area, (point_x, point_y), measureDist=False)
        cv2.polylines(frame, pts=[area], isClosed=True, color=(0, 0, 255))
        if polyTestRes >= 0:
            return True
    return False


FRAME_ANALYSIS_FREQ = 4
VIDEO_DIR = "vid"
JSON_DIR = "jsons"
USED_CLASSES = [2, 5, 7]  # 2 - car, 5 - bus, 7 - truck

init_keys = ["file_name", "car", "quantity_car", "average_speed_car", "van", "quantity_van", "average_speed_van", "bus",
             "quantity_bus", "average_speed_bus"]
result_df = pd.DataFrame(dict.fromkeys(init_keys, []))
print(result_df)

vids = os.listdir(VIDEO_DIR)
print("Доступные видео: ", vids)
for vid_fname_idx in range(len(vids)):
    vid_fname = vids[vid_fname_idx]
    json_name = vid_fname.split('.mp4')[0]

    with open(f'{JSON_DIR}/{json_name}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = YOLO('yolov8x.pt')

    vid_path = f"{VIDEO_DIR}/{vid_fname}"
    vid_capture = cv2.VideoCapture(vid_path)
    vid_fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print(f"=== {vid_fname} - {vid_fps} FPS ===")

    appeared_ids = []

    transport_dict: dict = {}

    frame_idx = 0
    frame_idx_in_cond = 0

    while vid_capture.isOpened():
        success, frame = vid_capture.read()

        if success:
            if frame_idx % FRAME_ANALYSIS_FREQ == 0:
                frame = cv2.resize(frame, (640, 360)) # (1280, 720)

                width = frame.shape[1]
                x_min, x_max = get_crop_x(annotated_frame=frame, data=data)

                new_width = width - (x_min + x_max)
                start_x = x_min
                cropped_frame = frame[:, start_x:start_x + new_width]
                color = (0, 0, 255)

                # !!! Изменить verbose на True если нужна инфа от модели !!!
                track_results = model.track(frame,
                                            persist=True,
                                            classes=USED_CLASSES,
                                            verbose=False,
                                            conf=0.5,
                                            imgsz=(384, 640)) 
                annotated_frame = track_results[0].plot()
                try:
                    bb_on_frame_ids: np.ndarray = track_results[0].boxes.id.numpy()
                    bb_on_frame_cls: np.ndarray = track_results[0].boxes.cls.numpy()
                    bb_center: np.ndarray = track_results[0].boxes.xywh.numpy()
                    bb_corners: np.ndarray = track_results[0].boxes.xyxy.numpy()

                    for i in range(len(bb_on_frame_ids)):

                        idx = str(int(bb_on_frame_ids[i]))

                        if idx not in transport_dict.keys():
                            transport_dict[idx] = Transport(str(idx))

                        cv2.circle(annotated_frame, (int(bb_center[i][0]), int(bb_center[i][1] + (bb_center[i][3] / 2))), 8, (232, 88, 163), -1)

                        if is_inside_zone(bb_center[i][0], bb_center[i][1] + (bb_center[i][3] / 2), annotated_frame):
                            transport_dict[idx].frames.append(frame_idx_in_cond)
                            transport_dict[idx].ts_type.append(int(bb_on_frame_cls[i]))
                            transport_dict[idx].inside_area = True

                except Exception as e:
                    print(e)

                annotated_frame = cv2.resize(annotated_frame, (1280, 640))
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                key = cv2.waitKey(1)
                frame_idx_in_cond += 1

                if key == 27:
                    break
            frame_idx += 1
        else:
            break

    class_average_speed: dict = {}
    count_dict = dict.fromkeys(USED_CLASSES, 0)

    for k in transport_dict.keys():
        ts: Transport = transport_dict[k]

        if len(ts.frames) < 6:
            continue

        import operator

        d = dict.fromkeys(ts.ts_type)
        for k in d.keys():
            d[k] = ts.ts_type.count(k)
        ts_type = max(d.items(), key=operator.itemgetter(1))[0]

        if ts.inside_area:
            count_dict[ts_type] += 1

        if ts.inside_area:
            delta_f = ts.frames[-1] - ts.frames[0]
            print(f'{delta_f} = {ts.frames[-1]} - {ts.frames[0]}')
            m: float = 20
            ms_to_kmh: float = 3.6
            k: float = (vid_fps / FRAME_ANALYSIS_FREQ)
            average_speed = m / ((delta_f + 1) / k) * ms_to_kmh
            class_average_speed.setdefault(ts_type, []).append(average_speed)

    for k in class_average_speed.keys():
        class_average_speed[k] = sum(class_average_speed[k]) / len(class_average_speed[k])

    print(f"Средняя скорость по классу: {class_average_speed}")
    print(f"Финальное количество объектов в видео {vid_fname}: {count_dict}")
    vid_fname_formatless = vid_fname.split(".")[0]

    try:
        car_avg_spd = class_average_speed[2]
    except KeyError: car_avg_spd = 0
    try:
        bus_avg_spd = class_average_speed[5]
    except KeyError: bus_avg_spd = 0
    try:
        van_avg_spd = class_average_speed[7]
    except KeyError: van_avg_spd = 0

    data_inserted = [vid_fname_formatless, "car", count_dict[2], car_avg_spd, "van", count_dict[7], van_avg_spd, "bus", count_dict[5], bus_avg_spd]

    result_df.loc[vid_fname_idx] = data_inserted
    result_df.to_excel('./result/submission.xlsx', index=False)
    result_df.to_csv('./result/submission.csv', index=False)

    vid_capture.release()
    cv2.destroyAllWindows()

print(result_df)
