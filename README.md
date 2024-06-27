#  Face Recognition using Convnext Model

## Introduction
This repo is python source for training and inference to recognite who is this person, , we train and get the feature extracter to extract vector and compare with our database via webcame.

For your information, Triplet loss is a loss function used in Machine Learning, the goal of it is to learn an embedding space where similar instances are closer to each other, while dissimilar instances are farther apart. The loss funciton takes into account triplets of examples: an anchor, a positive example, and a negative example

<!-- <p align="center">
  <img src="demo/video-1.gif"><br/>
  <i>Sample result</i>
</p> -->

<p align="center">
    <img src="demo/demo-running-result.png", width="500"><br/>
    <i> Sample result </i>
</p>

## Dataset
List of each dictionary, the form is like: 
[
    {
        "name": person1, 
        "face_feature": face_feature_person1
    }
]

## How to use my code
With my code you can:
- Train the model: In progress...
- Test your trained model (or my model) by running `bash run.sh`


### Step by step:
- Runinng command `git clone https://github.com/PhuTd03/Multi-Face-Recognize.git`
- Change direction to folder you have clone: `cd face-recognition`
- Install libraries (Recommend to use env like conda): `pip install -r requirements.txt`

#### To training (In progress...)

#### To inference
*Note: You can use your custome  data by add its to list of dictionary and change with file databse.pickle in direction `data/database.pickle`*
- run file run.sh by command `bash run.sh` (If you use Linux)
- Open the server in your comamnd line



## Reference
I would like to express my graditude to @ahmedbadr97 for creating the increadible repository [conv-facenet](https://github.com/ahmedbadr97/conv-facenet), from which most of this code is derived. Thank you very much!