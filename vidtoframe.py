import cv2
import os
import numpy


def vid_to_frame(video_path, output_dir):
    '''
    Input:
    video_path: string, path of video
    output_dir: string, path of output

    Output:
    string, details on # of frames, the video, and shape of the frames
    '''
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    while True:
        success, frame = cap.read() #success is if a frame is successfully capture, frame is an np.ndarray of the image
        if not success:
            break
        frame_filename = f"{output_dir}/{video_path}_frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        frame_shape = frame.shape
    cap.release() #tells the system we are done using the video
    return('generated',frame_count,'frames from video',video_path,'with frame size', frame_shape)


#test, cd to video folder first
print(vid_to_frame('testvid.MOV','testframes'))

