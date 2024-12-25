import glob
import tqdm
import os
import cv2
import numpy as np


# DATASET = 'hockey'
# videos = glob.glob(f'../datasets/{DATASET}_dataset/original_data/*/*.avi', recursive=True)

DATASET = 'RWF-2000'
videos = glob.glob(f'{DATASET}/original_data/val/*/*.avi', recursive=True)

GAMMA = 0.67  # if GAMMA < 0, no gamma correction will be applied
# Precompute gamma correction table
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]).astype("uint8")


def process_gamma(video_name):
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f"{video_name.replace('original_data', 'gamma_data')[:-4]}.avi", fourcc, fps, (width,  height))
    output_path = video_name.replace('original_data', 'gamma_data')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(f"{output_path[:-4]}.avi", fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame or end of video reached")
            break
        frame = cv2.LUT(frame, gamma_table)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print ("Start Prepocessing ...")
    for video_path in tqdm.tqdm(videos):
        # print ("Process gama ...")
        # if GAMMA > 0:
        #     process_gamma(video_path)

        video_gamma = video_path.replace('original_data', 'gamma_data') if GAMMA > 0 else video_path
        if not os.path.exists(video_gamma):
            print(f"File not found: {video_gamma}")

        video_save_path = video_path.replace('original_data', 'openpose_gamma_bg')
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        print ("Extract Pose")
        # TODO: use openpose Python API for faster processing
        os.system(f'cd openpose && ./build/examples/openpose/openpose.bin --video ../{video_gamma[:-4]+".avi"} --display 0 --write_video ../{video_save_path[:-4]+".avi"} > /dev/null')
        print ("Done !!!")
    
    print ("Finished!!!")