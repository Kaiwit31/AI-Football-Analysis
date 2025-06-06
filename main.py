from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovement
from view_transformer import viewTransformer
from speed_and_distance import SpeedAndDistance

def main():
    # read video
    video_frames = read_video('input_video/08fd33_4.mp4')
    
    if not video_frames:
        print("Unable to read the video or the video has no frames.")
        return
    print(f"Video loaded: {len(video_frames)} เฟรม")

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Get tracks 
    tracks = tracker.get_object_tracks(video_frames,
                                   read_from_stub=False,
                                   stub_path='stubs/track_tubs.pkl')
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement = CameraMovement(video_frames[0])
    camera_movement_per_frame = camera_movement.get_camera_movement(video_frames,
                                                                        read_from_stub=True,
                                                                        stub_path='stubs/camera_movement_stub.pkl')
    camera_movement.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    #view Transformer
    view_transformer = viewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks) 
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance
    speed_and_distance = SpeedAndDistance()
    speed_and_distance.add_speed_and_distance_to_tracks(tracks)
    
    # save cropped image of a player
    #for track_id, player in tracks['players'][0].items():
        #bbox = player['bbox']
        #frame = video_frames[0]
        #   
        #   #crop bbox from frame 
        #cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #  
        #  # save the cropped image
        #cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        #     
        #break
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]
            
            
    # Assign Ball Aqisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_tracks in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_tracks, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)    
    
    ## Draw camera movement
    output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Check_video_frames
    output_video_frames = [f for f in output_video_frames if f is not None]

    ## Draw Speed and Distance
    speed_and_distance.draw_speed_and_distance(output_video_frames,tracks)
    
    # save video
    if not output_video_frames:
        print("Not Frames")
    else:
        print("Save Video...")
        save_video(output_video_frames, 'output_videos/output_video.avi')
        print("Done! The video has been saved.")

if __name__ == "__main__":
    main()
