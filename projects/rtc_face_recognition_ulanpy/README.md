# Real Time Face Recognition System based on pre-trained FaceNet 

This repo could be used for RTC systems like WebRTC or RTCP. Due to the heavy load of face recognition models, the project was optimized in computational complexity providing good performance in embedded systems devices like Jetson Orin from Nvidia 

![face_recognition]([https://sun9-16.userapi.com/s/v1/ig2/g-FLtkQjGr8CUdHXxB6KbTj6cEAdgD23PbXsUrvO5yjk8E9wT1H7UE53AxFXaMgmgsP4sd8wyfA00ymkB563LMUD.jpg?quality=95&as=32x43,48x64,72x96,108x144,160x213,240x320,360x480,480x640,540x720,640x853,720x960,960x1280&from=bu&u=8kcZOkhgQwK7o1q8d5iZ9mUqlNxnDn5kpsoYOUZM4mk&cs=303x404](https://sun9-46.userapi.com/impg/sFoNRMtR4XCNLZjpYoJrE1DIrenmsu3Of4K-SQ/OenUAZLI1wU.jpg?size=640x516&quality=95&sign=dd94159bcdb5a3994d77d19178dcfee2&type=album))
## Installation

- Create your custom dataset consisting of .jpg images in dataset/person1 and so on dirs. Remove 1.txt files from dir

- Open the Facenet_and_missforest_paper file to train your model on your custom dataset. The project uses MTCNN architecture to detect faces -> pass them as 160x160x3 input tensor FaceNet() embedder -> create hyperplane with SVM to classify obtained embeddings from the dataset. The training file also makes occlusions and image augmentations. You could try to play with training parameters

- Move faces_embeddings_new1.npz, svm_model_new1.pkl, label_encoder_new1.pkl weights to src/models dir

- Install project dependencies with poetry on the project project dir
```
poetry update
```
- Run main.py
```
python main.py
```

Any questions DM @kamikadze24 on telegram

## Future Improvments 

- Implement Local Vector DB to store embeddings instead of SVM classifier
- Automate proccess of adding/removal of person
- Use advanced Face Tracking system like DeepSort
