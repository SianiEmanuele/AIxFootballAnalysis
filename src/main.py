from utils import read_video, save_video

def main():
    video_frames = read_video(r"../input_videos/08fd33_4.mp4")
    print(video_frames)


    save_video(video_frames, r"../output_videos/output.avi")


if __name__ == '__main__':
    main()