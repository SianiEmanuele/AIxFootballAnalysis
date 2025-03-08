from dotenv import load_dotenv
from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video(r"input_videos\08fd33_4.mp4")

    tracker = Tracker(model_path=r"src\training\yolo_football_analysis\yolo9c_dataset_v2\weights\best.pt")


    tracks = tracker.get_object_tracks(video_frames)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    save_video(output_video_frames, r"output_videos/yolo9c_v2.avi")


if __name__ == '__main__':
    main()