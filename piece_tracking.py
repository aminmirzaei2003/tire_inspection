
import torch
import cv2
from PIL import Image
from romatch import roma_outdoor
import time

import numpy as np
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

# def feature_matching(frame,piece_im,):
#     w1,h1 = piece_im.size
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     w2 = frame.shape[1]
#     h2 = frame.shape[0]
#     # Convert to PIL Image
#     pil_image = Image.fromarray(frame_rgb)
        
#     warp, certainty = roma_model.match(piece_im,pil_image, device=device)
#     new_warp = warp[certainty>0.90]
#     kpts_1, kpts_2 = roma_model.to_pixel_coordinates(new_warp, h1, w1, h2, w2)
    
#     return kpts_1,kpts_2

def feature_matching(img,piece_ten):
    image2 = img  
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    img2_tensor = frame2tensor(image2_gray, 'cuda')
    pred = matching({**piece_ten, 'image1': img2_tensor})
    kpts0 = piece_ten['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].detach().cpu().numpy()
    
    valid = (matches > -1) & (confidence>0.35)   
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0,mkpts1

def piece_pose(kpts_1,kpts_2,bias):
    w1,h1 = bias
    position = np.mean((kpts_2-kpts_1),axis=0)
    position[0] = position[0]+w1
    position[1] = position[1]+h1

    return position

def point_in_box(point,box):
    box_x,box_y,box_w,box_h = box
    in_box = (((box_x+box_w)>=point[0]) * (point[0]>=box_x) * ((box_y+box_h)>=point[1]) * (point[1]>=box_y))

    return in_box



# piece1_im = Image.open('piece_6.png')
# piece2_im = Image.open('piece_2.png')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# roma_model = roma_outdoor(device=device)

# Path to your video
video_path = "data/videos/00-46-b8-08-2b-81-01-01-20250729-113930_10.mp4"  # change this to your video path

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

i = 0
frequency = 1

box1 = (270,130,45,45)
box2 = (323,165,52,55)

box3 = (330,80,100,100)

threshold1 = 7
threshold2 = 7
threshold3 = 7

flag1_1 = False
flag2_1 = False

flag_m = False

flag2 = False

counter = 0
counter_1 = 0
text = ''
color = (0,0,0)
text_1 = ''
color_1 = (0,0,0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
output_path = "results/videos/00-46-b8-08-2b-81-01-01-20250729-113930_12_blader.mp4"  # change this to your video path
# Create VideoWriter object

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.0001, 'max_keypoints': -1},
    'superglue': {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.25}
}

matching = Matching(config).eval().to('cuda')
keys = ['keypoints', 'scores', 'descriptors']

image1 = cv2.imread('piece_6.png')  # Load the image
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img1_tensor = frame2tensor(image1_gray, 'cuda')
piece1_ten = matching.superpoint({'image': img1_tensor})
piece1_ten = {k+'0': piece1_ten[k] for k in keys}
piece1_ten['image0'] = img1_tensor

image2 = cv2.imread('piece_2.png')  # Load the image
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
img2_tensor = frame2tensor(image2_gray, 'cuda')
piece2_ten = matching.superpoint({'image': img2_tensor})
piece2_ten = {k+'0': piece2_ten[k] for k in keys}
piece2_ten['image0'] = img2_tensor

box1_x,box1_y,box1_w,box1_h = box1
box2_x,box2_y,box2_w,box2_h = box2
box3_x,box3_y,box3_w,box3_h = box3

while True:    
    ret, frame = cap.read()
    
    if not ret:
        # No more frames, video has ended
        break
    if(i%frequency==0 and i>0):
        
        if(not flag_m):
            
            counter_1 = 0
            # w1,h1 = piece1_im.size
            w1 = image1.shape[1]
            h1 = image1.shape[0]
            area1 = frame[box1_y:box1_y+box1_h,box1_x:box1_x+box1_w]
            area2 = frame[box2_y:box2_y+box2_h,box2_x:box2_x+box2_w]
            start_time = time.time()
            kpts1_1, kpts1_2 = feature_matching(area1,piece1_ten)
            kpts2_1, kpts2_2 = feature_matching(area2,piece1_ten)
            print(len(kpts1_1),len(kpts2_1))
            print((time.time() - start_time) * 1000, 'ms')

            position_1 = piece_pose(kpts1_1,kpts1_2,(w1/2,h1/2))
            position_1[0] += box1_x
            position_1[1] += box1_y
            position_2 = piece_pose(kpts2_1,kpts2_2,(w1/2,h1/2))
            position_2[0] += box2_x
            position_2[1] += box2_y
            box1_in = point_in_box(position_1,box1) 
            box2_in = point_in_box(position_2,box2) 
            
            if(box1_in*(len(kpts1_2)>threshold1)):
                text = 'Piece 1 in position 1'
                color = (0,255,255)
                flag1_1 = True
            elif(box2_in*(len(kpts2_1)>threshold2)):
                if(counter>0 and flag1_1):
                    text_1 = 'Wrong start detection!'
                    color_1 = (0,0,255)
                counter = 0
                text = 'Piece 1 in Position 2'
                color = (0,255,255)
                if(flag1_1):
                    flag2_1 = True
                    # flag1_1 = False
            else:
                if(flag2_1):
                    counter+=1
                    if(counter==1):
                        text_1 = 'Begin building ....'
                        text = ''
                        color_1 = (0,255,0)
                    if(counter>20):
                        flag2_1 = False
                        flag1_1 = False
                        flag_m = True
                        counter = 0
            if(len(kpts1_1)>threshold1):
                frame = cv2.circle(frame, (int(position_1[0]), int(position_1[1])), 5, (255,100,100), -1)
            if(len(kpts2_1)>threshold2):
                frame = cv2.circle(frame, (int(position_2[0]), int(position_2[1])), 5, (255,100,100), -1)                
        else:
            counter = 0
            w1 = image1.shape[1]
            h1 = image1.shape[0]
            area3 = frame[box3_y:box3_y+box3_h,box3_x:box3_x+box3_w]
            start_time = time.time()
            kpts3_1, kpts3_2 = feature_matching(area3,piece2_ten)
            print(len(kpts3_1))
            print((time.time() - start_time) * 1000, 'ms')
            position_3 = piece_pose(kpts3_1,kpts3_2,(w1/2,10))
            position_3[0] += box3_x
            position_3[1] += box3_y
            box3_in = point_in_box(position_3,box3)
            if(box3_in*(len(kpts3_1)>threshold3)):
                if(counter_1>0):
                    
                    text_1 = 'Wrong end detection!'
                    color_1 = (0,0,255)
                counter_1 = 0
                text = 'Piece 2 in Position 2'
                color = (0,255,255)
                flag2 = True

            else:
                if(flag2):
                    counter_1+=1
                    if(counter_1==1):
                        text_1 = 'End building!'
                        text = ''
                        color_1 = (255,0,0)
                if(counter_1>20):
                    flag2 = False
                    flag_m = False
                    

            ##

            if(len(kpts3_1)>threshold3):
                frame = cv2.circle(frame, (int(position_3[0]), int(position_3[1])), 5, (255,100,100), -1)
        # for j in range(len(kpts_1)):
        #     frame = cv2.circle(frame, (int(kpts_2[j][0]), int(kpts_2[j][1])), 1, (100,255,100), -1)
            ###
            # cv2.rectangle(frame, (box1_x2, box1_y2), (box1_x2 + box1_w2, box1_y2 + box1_h2), color=(0, 255, 0), thickness=2)
            # cv2.rectangle(frame, (box2_x2, box2_y2), (box2_x2 + box2_w2, box2_y2 + box2_h2), color=(0, 255, 0), thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    cv2.rectangle(frame, (box1_x,box1_y), (box1_x + box1_w, box1_y +box1_h), color=(0, 255, 0), thickness=2)
    cv2.rectangle(frame, (box2_x,box2_y), (box2_x + box2_w, box2_y +box2_h), color=(0, 255, 0), thickness=2)
    cv2.rectangle(frame, (box3_x,box3_y), (box3_x + box3_w, box3_y +box3_h), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, text, (20,40), font, font_scale, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text_1, (20,20), font, font_scale, color_1, 2, cv2.LINE_AA)
    cv2.imshow('Image', frame)
    # out.write(frame)
    # Wait 25 ms between frames, exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    i+=1

# Release the capture and close the window
cap.release()
out.release()
cv2.destroyAllWindows()
