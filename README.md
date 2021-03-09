# Logistic_Regression_of_Building_and_Ground
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
+-- .gitignore
+-- Logistic_Regression.py
+-- main.py
+-- make_training_data_v2.py
+-- README.md
+-- requirements.txt
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

<img src="/_res/Rogistic_Regression_Parameter.jpg" width="40%" height="40%" title="Logistic_Regression_Parameter" alt="Logistic_Regression_Parameter"></img>

To run code
```
python main.py
```

## Requirements
I uploaded requirements.txt in case,
Important Requirements are below
```
numpy==1.16.6
opencv-python==4.5.1.48
parmap==1.5.2
tensorflow-gpu==1.14.0
tqdm==4.56.0
```

## Conclusion
It doesn't work well. I have some results and questions and thoughts for a week
### 1 : not good feature
#### Using np.mean
<img src="/_res/mean_value.jpg" width="60%" height="60%" title="mean_value" alt="mean_value"></img>  
I use np.mean but it can't represent that it is building or ground.  
because in same value, it shows 2 labels. That mean it's not good feature  
#### Using with 768 dataset
<img src="/_res/768_result_plot.jpg" width="60%" height="60%" title="768_result_plot" alt="768_result_plot"></img>  
#### Using with 576,000 dataset
<img src="/_res/576000_result_plot.jpg" width="60%" height="60%" title="576000_result_plot" alt="576000_result_plot"></img>  

np.mean is not good way to get nice feature so i'm gonna find other way.  
Training filter's weights. If we can make good filter that seperating features well,  
Accuracy will rise.  
<img src="/_res/good_feature.jpg" width="60%" height="60%" title="good_feature" alt="good_feature"></img>

### 2 : not many building data
<img src="/_res/small_building_data.jpg" width="60%" height="60%" title="small_building_data" alt="small_building_data"></img>  
We have only 25% building data.  
I think it's not good for training.
I will preprocessing to make 5:5 ratio of building and ground dataset.  
but make function that finding good feature is first.
