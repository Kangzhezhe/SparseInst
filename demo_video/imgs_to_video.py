import glob
import os
import cv2
from tqdm import tqdm

def merge_image_to_video(folder_name):
    fps = 10
    firstflag = True
    file_list = sorted(os.listdir(folder_name))
    for f1 in tqdm(file_list):
        filename = os.path.join(folder_name, f1)
        frame = cv2.imread(filename)
        if firstflag == True:  
            firstflag = False
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            img_size = (frame.shape[1], frame.shape[0])
            video = cv2.VideoWriter("output_cats.mp4", fourcc, fps, img_size)
        frame_suitable = cv2.resize(frame, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
        video.write(frame_suitable)
    video.release()

if __name__ == '__main__':
    folder_name = "result_pandas"
    merge_image_to_video(folder_name)