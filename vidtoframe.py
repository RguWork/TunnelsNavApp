import cv2
import os
import re
import collections
import time

FRAME_RATE = 60
SECONDS_TO_SKIP_AT_VIDEO_EDGES = 1
FRAMES_TO_SKIP_AT_VIDEO_EDGES = FRAME_RATE * SECONDS_TO_SKIP_AT_VIDEO_EDGES
BLDG = r"([EB]\d+)"
HALL = r"([A-Z]\d+)"
DIRECTION = r"(NorthEast|SouthEast|NorthWest|SouthWest|North|South|East|West|)"
TITLE_FORMAT = BLDG+HALL+"_"+DIRECTION

def valid_video_title(title):
    '''
    Returns the generalized video title (without the trial number)
    Input:
    title: string, video title in data format

    Output:
    string, title without the sample number at the end. 
        If the video title is in the form of X-NE-37.mp4 then it is the 37th recording of room X.
        Returning just X
    '''
    title_match = re.match(TITLE_FORMAT, title)
    if not title_match:
        raise Exception(f"Video title {title} is not correctly formatted")
    return title_match.group(0)

def verify_dataset_not_parsed(data_path):
    '''
    Errors if data_path exists or Train, Validation, Test subdirectories exist in it. Otherwise creates those subdirectories.

    Input:
    data_path: string/path, path of the directory containing the parsed dataset

    Output:
    None
    '''
    split_paths = list(map(lambda split: os.path.join(data_path, split), ["Train", "Validation", "Test"]))
    split_directory_exists = map(os.path.exists, list(split_paths))
    if any(split_directory_exists):
        raise Exception(f"A dataset possibly exists at {data_path}. Did not modify it")
    
    for split_path in split_paths:
        os.makedirs(split_path)

def verify_videos(video_path):
    '''
    Errors if a video in the dataset is not labeled correctly.

    Input:
    data_path: string/path, path of the directory containing the raw data

    Output:
    None
    '''
    for _ in scan_videos(video_path):
        pass

def scan_videos(video_path):
    '''
    Input:
    folder_path: string, path of directory containing data as videos

    Output:
    Iterator[str, VideoCapture]: iterator over each (video_name, video) in the folder
    '''

    with os.scandir(video_path) as folder_iterator:
        for folder in folder_iterator:
            if not folder.is_dir():
                continue
            with os.scandir(folder.path) as video_iterator:
                for video in video_iterator:
                    if not video.is_file():
                        continue
                    title = valid_video_title(video.name)
                    video = cv2.VideoCapture(video.path)
                    yield title, video
                    video.release()

def total_frames_per_room(video_iterator):
    '''
    Input:
    video_iterator: Iterator[(str, VideoCapture)], iterator over videos of a (room, direction) with possible multiple samples                 

    Output:
    Dict[str, int], Map mapping the name of each (room, direction) -> total number of frames captured across all videos in the iterator
                        EX) Two videos v1, v2 are recorded for (B1-H1-N) in the iterator. (B1-H1-N) maps to the total frames in v1 and v2
    '''
    frame_map = collections.defaultdict(int)

    for title, video in video_iterator:
        frame_map[title] += video.get(cv2.CAP_PROP_FRAME_COUNT) - FRAMES_TO_SKIP_AT_VIDEO_EDGES
    
    print("min number of frames is", min(frame_map.values()))
    return frame_map

def downsample_frames(video_iterator, room_frame_totals):
    '''
    Input:
    video_iterator: Iterator[(str, VideoCapture)], iterator over videos of a (room, direction) with possible multiple samples
    room_frame_totals: Map[str, int], mapping from (room, direction) to number of frames belonging to it in video_iterators' videos.
                        
    Output:
    Iterator[(str, MatLike, int, int)], iterator over each "room name", image frame pair. 
                                        Note that "room name" is the same across each videos' frames.
                                        The last two integers are the index of the frame (excluding edges)
                                        and the number of valid frames (excluding edges). 
    '''
    minimum_frame_count = int(min(room_frame_totals.values()))
    for title, video in video_iterator:
        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        room_frame_count = int(room_frame_totals[title])

        frames_per_sample = room_frame_count / minimum_frame_count
        
        #Skip frames at video start
        for i in range(0, FRAMES_TO_SKIP_AT_VIDEO_EDGES):
            video.grab()

        last_valid_frame = video_frame_count - FRAMES_TO_SKIP_AT_VIDEO_EDGES
        # Code to skip frames, only yielding the correct portion of frames from the video 
        next_frame_to_capture = 0
        for frame_index in range(0, last_valid_frame):
            success = video.grab()
            if not success:
                raise Exception(f"Failed to grab frame {frame_index} from video {title}")
            
            if frame_index != int(next_frame_to_capture):
                continue
            frame = None
            frame_retrieved, frame = video.retrieve()
            if not frame_retrieved or frame is None:
                raise Exception(f"Failed to retrieve frame {frame_index} from video {title}")
            
            next_frame_to_capture += frames_per_sample
            yield (title, frame, frame_index, last_valid_frame)

def upload_frame_to(frame, name, split):
    '''
    Input:
    frame: MatLike, image frame
    name: str, location name, (room, direction), of the frame
    split: str, determining if the frame is to be uploaded to the Test, Train, or Validation folder
                        
    Output:
    None, Errors if frame failed to upload
    '''
    output_dir = os.path.join("Data", split, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = os.path.join(output_dir, f"{time.time_ns()}.png")
    if os.path.exists(image_path):
        raise Exception("Trying to overwrite an image frame that already exists")
    
    frame_uploaded = cv2.imwrite(image_path, frame)
    if not frame_uploaded:
        raise Exception("Failed to upload a frame on", name)

def upload_split_sequentially(frame_iterator, train_percentage, validation_percentage, testing_percentage):
    '''
    Input:
    frame_iterator: Iterator[(str, MatLike)], Iterator over (frame name, frame) pairs
    *_percentage: float, percentage of frames to be part of that split of the dataset
                        
    Output:
    None, Errors if percentages don't sum to 1
    '''
    if train_percentage + validation_percentage + testing_percentage != 1:
        raise Exception("Train, validation, testing splits don't add to 1, got:", train_percentage, validation_percentage, testing_percentage)
    
    for name, frame, frame_index, total_frames in frame_iterator:
        position =  frame_index / total_frames

        split = "Train" if position < train_percentage else \
            "Validation" if position < train_percentage + validation_percentage else \
            "Test"

        upload_frame_to(frame, name, split)

def upload_dataset(split_data_path, video_data_path, train_percentage, validation_percentage, testing_percentage):
    verify_dataset_not_parsed(split_data_path)
    verify_videos(video_data_path)

    video_it = scan_videos(video_data_path)
    tfpr = total_frames_per_room(scan_videos(video_data_path))
    frame_it = downsample_frames(video_it, tfpr)
    
    upload_split_sequentially(frame_it, train_percentage, validation_percentage, testing_percentage)

upload_dataset("Data", "Videos", 0.7, 0.15, 0.15)

<<<<<<< HEAD
    


#test, cd to video folder first
# print(vid_to_frame('CV_Vids/B6/B6H3_NorthRight.MOV','testframes'))
print(vid_to_frame('B6H3_NorthRightcopy', '/Users/rgu/Desktop/CV Final Project/data/B6H3_NorthRightcopy.MOV','/Users/rgu/Desktop/CV Final Project/data/trainframes','/Users/rgu/Desktop/CV Final Project/data/validframes','/Users/rgu/Desktop/CV Final Project/data/testframes'))
=======
>>>>>>> fd6465b276ce44f91f80dfb167c3933f37b64f4d
