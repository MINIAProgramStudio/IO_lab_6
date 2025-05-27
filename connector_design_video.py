if __name__ == '__main__':
    import sys
    from video_grayscale_to_rgb import create_video_from_gray_to_rgb as cvgr
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    AGENT1_NAME = sys.argv[3]
    AGENT2_NAME = sys.argv[4]

    print(input_path, output_path, AGENT1_NAME, AGENT2_NAME)
    cvgr(input_path, output_path, AGENT1_NAME, AGENT2_NAME)