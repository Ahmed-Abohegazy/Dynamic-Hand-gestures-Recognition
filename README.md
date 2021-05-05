Computer become a necessity and critical part of our daily life. Human and computer interaction is carried out through
traditional input tools such as the mouse, keyboard and other conventional, Hand gestures is going to be a valuable tool for HCI to enhance usability by establishing new methods for the interaction process.
Also, in the field of automotive user interfaces, touchless hand gesture recognition systems are becoming relevant, as they improve safety and comfort.
The objective is to develop a real-time system that takes an input from a camera then predicts the gesture class(static, dynamic), By using the power of CNN to develop a model and train it on a dataset of human gestures and increasing the performance to reach a high accuracy without neglecting the performance.
Instead of analyzing the hand shape separately in each image, we can analyze the hand movement in time, which means
analyzing multiple images at once. If you lift two fingers, it will be identified to the machine but the 3D CNN + LSTM gives us
more than that, it would be able to sense whether we push the two fingers left or right or, which also lets us observe the motion and the hand gesture




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
