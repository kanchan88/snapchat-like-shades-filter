# Snapchat Like Sunglass Filter

Here, I have the kaggle data with the images of people and the points in which their eyes are located. Lets say I have 96*96 pixel image of person and I am given the X,Y co-ordinate where there can be eyes. I am unable to push this data to git. Download it from here:https://www.kaggle.com/c/facial-keypoints-detection/data

I then trained this image using CNN, which in return gives me the model which can predict where the eyes of people can be in the given image of people. After getting the co-cordinate of eye position, I added shades over these pixels. Here, I have also used frontal face cascade to send only facial image to CNN model.

## Used
The code is in Python (version 3.6 or higher). You also need to install OpenCV and Keras libraries.

## Execution
Order of Execution is as follows:

Step 0 - Download the _'training.zip'_ file from [here](https://www.kaggle.com/c/facial-keypoints-detection/data) and extract it into the _'data'_ folder.

Step 1 - Execute ``` python model_builder.py ```

Step 2 - This could take a while, so feel free to take a break.

Step 3 - Execute ``` python shades.py ```
