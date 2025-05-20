# weapon-detection
Pothole Detection inference

# Weight

'''
https://drive.google.com/drive/folders/1HmBHd4mXZMrevKnuVUN0IsLrwoZOUxMB?usp=sharing
'''

# Run

'''
sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device /dev/video0:/dev/video0 -p 8000:8000 pothole-detection
'''

