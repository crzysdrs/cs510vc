CS510 - Introduction to Visual Computing - Motion Tracking
---------------

[CS510 Visual Computing](http://web.cecs.pdx.edu/~fliu/courses/cs410) - [Project Report](https://dl.dropboxusercontent.com/u/13119212/CS510vc-MitchSouders-proj2.pdf)

This project is released under the GPLv2 as described in [LICENSE](LICENSE)

Usage
--------------

This project is implemented in Python and OpenCV to produce a tool which can track multiple objects moving in a scene.

    usage: tracking.py [-h] [--max_size MAX_SIZE] [--learning_rate LEARNING_RATE]
                       [--mask_shadow] [--debug] [--output OUTPUT]
                       video

To run it all you need is a video that was filmed with a stationary camera. It's not a perfect motion tracker but it will work well on traffic scenes with visible spacing between vehicles/pedestrians.

Example
----------
    ./tracking.py input_video.mp4 --output output_video.mp4


