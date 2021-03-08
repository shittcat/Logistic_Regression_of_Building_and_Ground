# Rogistic_Regression_of_Building_and_Ground
It analyze patterns between ground and building

## Directory Tree
```
.
+-- _res
|   +-- Rogistic_Regression_Parameter.jpg
+-- train
|   +-- ground
|   +-- image
+-- test
|   +-- ground
|   +-- image
+-- Logistic_Regression.py
+-- main.py
+-- make_training_data_v2.py
```

## Introducton
#### make_training_data_v2.py
make_training_data_v2.py makes dataset.
It make 256 x 256 image to 16 x 16 RGB images, and label of each images.
if 16 x 16 RGB image's segmented image have more than 128 white pixels, then label is 1 or label is 0
It means if label is 1, that 16 x 16 RGB image is building image and if not, that is ground image
it use Multiprocessing. It returns datasets of 16 x 16 RGB images, and labels of them

#### Logistic_Regression.py
It is simple Logistic_Regression.
It gets dataset, and make input_data to batch dataset

## How to use
Input Train data and Testdata in each 'train' and 'test' folder.
This code need RGB image and segmented image of it.

![Rogistic_Regression_Parameter](/_res/Rogistic_Regression.jpg "Optional title")
<img src="/_res/Rogistic_Regression.jpg" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Rogistic_Regression_Parameter"></img>
  
