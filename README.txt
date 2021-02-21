IMPORTANT NOTES 
### The technologies used:
Python 3.7  
TensorFlow GPU V2
KERAS 
OpenCV
pynput
Spyder Ide 

### Description to read before you begin:
-extrecting_training_samples.PY: due to  the limitation of the processing power this script is used to extract wanted gestures and the num of samples per gesture 
-test_app.py: is used to start the cam and predict gesture 
-controll_commands.py: the script is used to link between the gesture predicted and the command to get applied 
-train.py: to train models 
-plots_traindata.py: is imported while taring to plot training resutls 
-training_samples folder : containg subset form 20 bn jester data set which is used for the training purpose 
-resultfolder : after traing the code genrates txt file contains information about every epoch in the traing proccess
-saved_modelfolder : usign callbacks ,best models wiehgts are saved into this folder while training 

###the subset of the 20 bn jester data set link in oder to rain the model 
https://drive.google.com/drive/folders/1D47fhHbNIHcgcPAhqo9z-MfdNytvVAEV?usp=sharing


###how to run the code to control Spotify :
-make sure Spotify app is already downloaded if not download it using windows 10 Microsoft store 
-run test_app.py
-wait until the webcam launch 
-open Spotify (note make sure Spotify is selected and in full-screen mode)
- make sure you click on the area of spotify controlers  on the botthom 

geasture and respoding controls :
thumbs up > increase volume 
thumb down > reduce volume 
Drumming fingers > play/stop
swipe to finger left >prevoius track 
swipe to fingers right > next track
