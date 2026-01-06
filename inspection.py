import cv2
import os
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
from romatch import roma_outdoor
import time
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def feature_matching(frame,piece_im):
    w1,h1 = piece_im.size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w2 = frame.shape[1]
    h2 = frame.shape[0]
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
        
    warp, certainty = roma_model.match(piece_im,pil_image,device=device)
    new_warp = warp[certainty>0.9]
    kpts_1, kpts_2 = roma_model.to_pixel_coordinates(new_warp, h1, w1, h2, w2)
    
    return kpts_1,kpts_2

# def feature_matching(img,piece_ten):
#     image2 = img  
#     image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     img2_tensor = frame2tensor(image2_gray, 'cuda')
#     pred = matching({**piece_ten, 'image1': img2_tensor})
#     kpts0 = piece_ten['keypoints0'][0].cpu().numpy()
#     kpts1 = pred['keypoints1'][0].cpu().numpy()
#     matches = pred['matches0'][0].cpu().numpy()
#     confidence = pred['matching_scores0'][0].detach().cpu().numpy()
    
#     valid = (matches > -1) & (confidence>0.25)   
#     mkpts0 = kpts0[valid]
#     mkpts1 = kpts1[matches[valid]]

#     return mkpts0,mkpts1

def piece_pose(kpts_1,kpts_2,bias):
    w1,h1 = bias
    position = torch.mean((kpts_2-kpts_1),dim=0)
    position[0] = position[0]+w1
    position[1] = position[1]+h1

    return position

def point_in_box(point,box):
    box_x,box_y,box_w,box_h = box
    in_box = (((box_x+box_w)>=point[0]) * (point[0]>=box_x) * ((box_y+box_h)>=point[1]) * (point[1]>=box_y))

    return in_box

def depth_points(points,depth):
    points = np.array(points)
    depth_points = np.mean(depth[(points[:,1]).astype(int),(points[:,0]).astype(int)])
    return depth_points

def body_pose(frame):
    hands_right = []
    hands_left = []
    body_points = []
    results = yolo_model.predict(frame, verbose=False)
    for result in results:
        keypoints = result.keypoints.xy  # shape: (num_people, 17, 2)

        for person_kpts in keypoints:
            left_wrist = person_kpts[9].tolist()   # [x, y]
            right_wrist = person_kpts[10].tolist() # [x, y]
            body_points.append(person_kpts)
            hands_left.append(left_wrist)
            hands_right.append(right_wrist)

            # Draw them on frame
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 0, 255), -1)
    
    return hands_left,hands_right,body_points


device = 'cuda' if torch.cuda.is_available() else 'cpu'
roma_model = roma_outdoor(device=device)

# Load YOLO pose model (you can use yolov8n-pose.pt for faster but less accurate results)
yolo_model = YOLO("yolo11x-pose.pt")  # or yolov8n-pose.pt, yolov8m-pose.pt

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

model_depth = DepthAnythingV2(**model_configs[encoder])
model_depth.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
model_depth = model_depth.to(device).eval()
# Open video
video_path = "data/videos/00-46-b8-08-2b-81-01-01-20250729-113930_10.mp4"  # your video file

cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
piece_im = Image.open('piece_4.png')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
output_path = "results/videos/00-46-b8-08-2b-81-01-01-20250729-113930_10.mp4"  # change this to your video path
# Create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# box = (180,85,55,55)
text = ''
color = (0,0,0)
text_1 = ''
color_1 = (0,0,0)
box = (176,81,59,59)
box1_x,box1_y,box1_w,box1_h = box
i = 0 
frequency = 7
threshold = 50
distance_threshold1 = 0.25
distance_threshold2 = 0.5
flag = False
box1_in = False
counter_1 = 0
depth_hand = 0
depth_difference = 0.75
my_kpts = 0
my_flag = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if(i%frequency==0 and i>650):
        my_flag = True
        text = ''
        text_1 = ''
        w1,h1 = piece_im.size
        start_time = time.time()
 
        # area1 = frame[box1_y:box1_y+box1_h,box1_x:box1_x+box1_w]
        kpts_1, kpts_2 = feature_matching(frame,piece_im)
        print(len(kpts_1))
        position = piece_pose(kpts_1,kpts_2,(w1/2,h1/2))
        
        if(len(kpts_2)>threshold):
            my_kpts = kpts_2

        box1_in = ((point_in_box(position,box) and(len(kpts_2)>threshold)))

        ##
        # if(len(kpts_1)>0):
        #      frame = cv2.circle(frame, (int(position[0]), int(position[1])), 1, (255,100,100), -1)
        # for j in range(len(kpts_1)):
        #     frame = cv2.circle(frame, (int(kpts_2[j][0]+box1_x), int(kpts_2[j][1])+box1_y), 3, (100,255,100), -1)
        ##
        hands_left,hands_right,body_points = body_pose(frame)
        depth = model_depth.infer_image(frame)
        if(box1_in):
            text_1 = 'Tire on stand'
            color_1 = (0,255,255)
            flag = True
            counter = 0
            counter_1 = 0
        else:
            if(flag):
                
                depth_obj = depth_points(my_kpts.cpu().long(),depth)
                
                x,y,w,h = box
                depth_box = depth_points(np.array([range(x,x+w),range(y,y+h)]),depth)
                counter+=1
                if((depth_box-depth_obj)>depth_difference):
                    counter_1+=1
                if(counter_1>1 or counter>4):
                    flag = False
        # print('Flag:',flag)
        for j in range(len(hands_right)):
            
            if((point_in_box(hands_right[j],box) or point_in_box(hands_left[j],box)) and flag and counter<4):
                depth_obj = depth_points(my_kpts.cpu().long(),depth)   
                # depth = model_depth.infer_image(frame) # HxW raw depth map in numpy
                body_depth = depth_points(body_points[j][6:12].cpu().long(),depth)
                
                
                if(np.abs(depth_obj-body_depth)<distance_threshold2):
                    print(body_depth,depth_obj)
                    print('Inspection...')
                    text = 'Inspection...'
                    color = (255,0,0)
        print((time.time() - start_time) * 1000, 'ms')
        # cv2.imshow('Depth',depth/20)
        x,y,w,h = box
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    if(text=='Inspection...'):
        overlay = frame.copy()
        cv2.rectangle(overlay, (160, 50), (270, 220), (255, 0, 0), -1)

        # Blend overlay with the original frame
        alpha = 0.25  # transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if(text_1=='Tire on stand' and text!='Inspection...'):
        overlay = frame.copy()
        cv2.rectangle(overlay, (box1_x,box1_y), (box1_x + box1_w, box1_y +box1_h), (0, 255, 255), -1)

        # Blend overlay with the original frame
        alpha = 0.25  # transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
 
    cv2.rectangle(frame, (box1_x,box1_y), (box1_x + box1_w, box1_y +box1_h), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, text, (20,20), font, font_scale, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text_1, (20,40), font, font_scale, color_1, 2, cv2.LINE_AA)
    if(my_flag):
        out.write(frame)
    cv2.imshow("Hands", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
         break 
    i+=1

cap.release()
out.release()
cv2.destroyAllWindows()

