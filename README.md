# Dynamic-Hand-gestures-Recognition
### Computer become a necessity and critical part of our daily life. Human and computer interaction is carried out through traditional input tools such as the mouse, keyboard and other conventional

    

>  - [ ] IMPORTANT NOTES

 - The technologies used:
> Python 3.7
> TensorFlow GPU V2
> KERAS
> OpenCV
> pynput  

>  - [ ] Description to read before you begin:

 - extrecting_training_samples.PY: 
 due to  the limitation of the processing power this script is used to extract wanted gestures and the num of samples per gesture.
 - test_app.py: is used to start the cam and predict gesture
 - controll_commands.py: 
 the script is used to link between the gesture predicted and the command to get applied 
- train.py: to train models
- plots_traindata.py: is imported while taring to plot training resutls 
- training_samples folder : 
-containg subset form 20 bn jester data set which is used for the training purpose 
- resultfolder : 
after traing the code genrates txt file contains information about every epoch in the traing proccess
- saved_modelfolder : 
usign callbacks ,best models wiehgts are saved into this folder while training


### the subset of the 20 bn jester data [Dataset](https://drive.google.com/drive/folders/1D47fhHbNIHcgcPAhqo9z-MfdNytvVAEV?usp=sharing) link in oder to rain the model.

> ### How to run the code to control Spotify :

- make sure Spotify app is already downloaded if not download it using windows 10 Microsoft store 
- run test_app.py
- wait until the webcam launch 
- open Spotify (note make sure Spotify is selected and in full-screen mode)
-  make sure you click on the area of spotify controlers  on the botthom

## Experiment  results
•The model is trained on training sample contains 1800 video clips for 5 different gestures.
•Model is a combination network between 3D CNN and LSTM
•The gestures are combination of 2 static gestures and 3 dynamic gesture.
•The model achieved 98% on the training accuracy and 90% on the validation accuracy.

![Accuracy](https://drive.google.com/file/d/110PTw-iwTl9W3hGngn1TCjaflCM3uF6d/view?usp=sharing)
![Loss](https://drive.google.com/file/d/1eBqF7B8m0ifUb-xwEMDN07whKCHkcWaG/view?usp=sharing)

 ![loss](learning%20curve%202.png)

 
