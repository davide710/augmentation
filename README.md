# augmentation

Twice the dataset, _same_ the time for annotation!

Multiply the size of the dataset (rotations, changes in brightness and contrast) in no time and don't waste another instant annotating it.

### Why?

Data augmentation is fundamental for a good object detection model. The problem is, the most boring part of object detection is annotating the dataset 
(e.g. using https://pypi.org/project/labelImg/ for YOLO model), and having to do it multiple times because for slightly different images feels like
something that should be done automatically. And so it is (from now on)!

### How?

Put your annotated images in a directory with the annotation (.txt) files. The code assumes that for the image "my_image.jpg" there is the file "my_image.txt", structured as follows: `class_id center_x center_y bounding_box_width bounding_box_height` (see examples in [resources/](resources/)), where `center_x` and `bounding_bow_width` are normalized to the image width and `center_y`, `bounding_box_height` to the image height.
Now:
```
git clone https://github.com/davide710/augment-it.git
cd augment-it
pip install numpy opencv-contrib-python
```
You can see a demo of how the code works by running `python3 demo.py`. This shows how the images (data I used for an actual object detection model) are rotated and how the bounding box automatically follows the object thanks to the magic of linear algebra. 

![original image](https://github.com/davide710/augment-it/assets/106482229/8c8181ff-c703-4a01-9911-e1048ef7e922)
![rotated image and box](https://github.com/davide710/augment-it/assets/106482229/9c50211b-3b6e-418f-a31e-e33bf6596cf6)

To use the functions in your code, simply
```
from rotation import rotate
from brightness_contrast import change_brightness_contrast

rotate(file_list, min_angle=5, max_angle=355, make_annotations=True, rotate_bbox=False, debug=False)
change_brightness_contrast(file_list, min_alpha=0.5, max_alpha=1.5, min_beta=-60, max_beta=60, make_annotations=True)
```
+ use `rotate_bbox=False` if your objects are circular (hence you just need to rotate the center)
+ use `make_annotations=False` if you just want the new images but not the new annotation files
+ `rotate` will save the output files in the directory "rotated/" and add "_rot" to all file names
+ `change_brightness_contrast` will save the output files in the directory "br_and_con/" and add "_bc" to all file names

#
#  
