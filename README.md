# lstm-cnn-project

model : https://drive.google.com/file/d/1qSiMLSTf37ZZC-zAUDi41fwsOmiECHcL/view?usp=sharing

---------
## using 
```
## model build
git clone
cd docker
bash docker_build.sh
bash docker_run.sh
docker exec -it geon-test.sh

## Download model
cd models
## put model
cd ..
exit
## xhost start
xhost +
docker exec -it geon-test bash
git clone https://github.com/geon0430/lstm-cnn-project.git
cd lstm-cnn-project
export DISPLAY=:1
python predict_web_video.py, result_video.py
```
## save_video 
------------

## inputs_inputs_Data
## 1. input video change path in result_video.py 2. datasets/1.mp4
```
2. Download model
   - model add
