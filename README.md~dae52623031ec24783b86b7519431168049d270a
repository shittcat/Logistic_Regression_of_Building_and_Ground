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

Change Rogistic_Regression_Parameter  
<img src="/_res/Rogistic_Regression_Parameter.jpg" width="40%" height="40%" title="Rogistic_Regression_Parameter" alt="Rogistic_Regression_Parameter"></img>

To run Rogistic_Regression
```
python main.py
```

## Requirments
```
absl-py==0.11.0
argon2-cffi==20.1.0
astor==0.8.1
astunparse==1.6.3
async-generator==1.10
attrs==20.3.0
backcall==0.2.0
bleach==3.3.0
cachetools==4.2.1
certifi==2020.12.5
cffi==1.14.5
chardet==4.0.0
colorama==0.4.4
cycler==0.10.0
decorator==4.4.2
defusedxml==0.6.0
entrypoints==0.3
flatbuffers==1.12
gast==0.2.2
google-auth==1.27.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
GPUtil @ file:///home/conda/feedstock_root/build_artifacts/gputil_1590646865081/work
grpcio==1.32.0
h5py==2.10.0
idna==2.10
importlib-metadata==3.4.0
ipykernel==5.5.0
ipython==7.21.0
ipython-genutils==0.2.0
ipywidgets==7.6.3
jedi==0.18.0
Jinja2==2.11.3
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==6.1.11
jupyter-console==6.2.0
jupyter-core==4.7.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver @ file:///C:/ci/kiwisolver_1612282618948/work
Markdown==3.3.3
MarkupSafe==1.1.1
matplotlib @ file:///C:/ci/matplotlib-suite_1613408055530/work
mistune==0.8.4
mkl-fft==1.2.1
mkl-random==1.1.1
mkl-service==2.3.0
nbclient==0.5.3
nbconvert==6.0.7
nbformat==5.1.2
nest-asyncio==1.5.1
notebook==6.2.0
numpy @ file:///C:/ci/numpy_and_numpy_base_1603468620949/work
oauthlib==3.1.0
olefile==0.46
opencv-python==4.5.1.48
opt-einsum==3.3.0
packaging==20.9
pandocfilters==1.4.3
parmap==1.5.2
parso==0.8.1
pickleshare==0.7.5
Pillow @ file:///C:/ci/pillow_1609786872067/work
prometheus-client==0.9.0
prompt-toolkit==3.0.16
protobuf==3.15.1
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycparser==2.20
Pygments==2.8.0
pyparsing @ file:///home/linux1/recipes/ci/pyparsing_1610983426697/work
pyrsistent==0.17.3
python-dateutil @ file:///home/ktietz/src/ci/python-dateutil_1611928101742/work
pywin32==300
pywinpty==0.5.7
pyzmq==22.0.3
qtconsole==5.0.2
QtPy==1.9.0
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7.1
scipy==1.6.1
Send2Trash==1.5.0
six @ file:///C:/ci/six_1605205426665/work
tensorboard==1.14.0
tensorboard-plugin-wit==1.8.0
tensorflow-estimator==1.14.0
tensorflow-gpu==1.14.0
termcolor==1.1.0
terminado==0.9.2
testpath==0.4.4
torch==1.7.1
torchaudio==0.7.2
torchvision==0.8.2
tornado @ file:///C:/ci/tornado_1606935947090/work
tqdm @ file:///tmp/build/80754af9/tqdm_1611857934208/work
traitlets==5.0.5
typing-extensions @ file:///home/ktietz/src/ci_mi/typing_extensions_1612808209620/work
urllib3==1.26.3
wcwidth==0.2.5
webencodings==0.5.1
Werkzeug==1.0.1
widgetsnbextension==3.5.1
wincertstore==0.2
wrapt==1.12.1
zipp==3.4.0
```
