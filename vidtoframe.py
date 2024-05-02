import cv2
import os
import numpy


def vid_to_frame(name, video_path, train_dir, val_dir, test_dir):
    '''
    Input:
    video_path: string, path of video
    output_dir: string, path of output

    Output:
    string, details on # of frames, the video, and shape of the frames
    '''
    cap = cv2.VideoCapture(video_path)
    name = video_path.split('.')[0]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #gets the total # of frames in the video

    train_end = int(total_frames * 0.7)
    val_end = int(total_frames * 0.85)

    output_dirs = {
        'train': os.path.join(train_dir, f"{name} train"),
        'val': os.path.join(val_dir, f"{name} val"),
        'test': os.path.join(test_dir, f"{name} test")
    }

    for dir in output_dirs.values():
        if not os.path.exists(dir):
            os.makedirs(dir) 

    frame_count = 0
    while True:
        success, frame = cap.read() #success is if a frame is successfully capture, frame is an np.ndarray of the image
        if not success:
            break

        if frame_count < train_end:
            set_type = 'train'
        elif frame_count < val_end:
            set_type = 'val'
        else:
            set_type = 'test'

        frame_filename = f"{output_dirs[set_type]}/{name}_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        frame_shape = frame.shape
    cap.release() #tells the system we are done using the video
    return('generated',frame_count,'frames from video',video_path,'with frame size', frame_shape)


    


#test, cd to video folder first
# print(vid_to_frame('CV_Vids/B6/B6H3_NorthRight.MOV','testframes'))
print(vid_to_frame('B6H3_NorthRightcopy', '/Users/rgu/Desktop/CV Final Project/data/B6H3_NorthRightcopy.MOV','/Users/rgu/Desktop/CV Final Project/data/trainframes','/Users/rgu/Desktop/CV Final Project/data/validframes','/Users/rgu/Desktop/CV Final Project/data/testframes'))
