# lstm-cnn-project

model : https://drive.google.com/file/d/1qSiMLSTf37ZZC-zAUDi41fwsOmiECHcL/view?usp=sharing

## using

```
xhost +
cd docker
bash docker_build.sh
bash docker_run.sh
docker exec -it geon-test.sh
git clone -b master https://github.com/geon0430/lstm-cnn-project.git
cd lstm-cnn-project
export DISPLAY=:1
bash run.sh
cd models
#가중치 넣기
cd ..
python predict_web_video.py
```
