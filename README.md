# hosbot-detection
Laying Down Position Detection using TensorFlow (EfficientDet01)


#conda environment
activate ob

#initialize anaconda environment


#nav to furniture-detection folder
cd furniture-detection

git clone https://github.com/tensorflow/models.git

#find the corresponding tensorflow config file in configs/tf2 folder (https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2)
#edit the config 
-Change the number of classes to number of objects you want to detect (4 in my case)
-Change fine_tune_checkpoint to the path of the model.ckpt file.
-Change fine_tune_checkpoint_type to detection
-Change input_path of the train_input_reader to the path of the train.record file:
-Change input_path of the eval_input_reader to the path of the test.record file:
-Change label_map_path to the path of the label map:
-Change batch_size to a number appropriate for your hardware, like 4, 8, or 16

#navigate to models folder
cd models/research

cd object_detection
python xml_to_csv.py

#in the generate_tfrecord.py change the labelmap to your respective labels and change the classes

#generate tf records
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

#create labelmap.pbtxt and put inside training folder
example:-
item {
    id: 1
    name: 'Raspberry_Pi_3'
}
item {
    id: 2
    name: 'Arduino_Nano'
}
item {
    id: 3
    name: 'ESP8266'
}
item {
    id: 4
    name: 'Heltec_ESP32_Lora'
}

#download protoc 3.4
#extract, find bin and copy protoc.exe into research folder

#Compile protos.
protoc object_detection/protos/*.proto --python_out=.

#copy tf2 setup.py into research folder
cd object_detection/packages/tf2

#pycoco tool problems (make sure visual studio build 14.0 ++ is installed)
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

#nav into research folder (install setup tf2)
python -m pip install .

#make sure in folder images ada file train & test which contains its respective photos and xml files


#start training!
python model_main_tf2.py \
    --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --model_dir=training \
    --alsologtostderr

#to get tensorflow graph nav to object_detection in another prompt
tensorboard --logdir=training/train

#stop training when loss is less than 0.2 or learning rate on tensorflow graph becomes stagnant 

#export inference graph (can make frozen graph for future inferencing)
python exporter_main_v2.py \
    --trained_checkpoint_dir=training \
    --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --output_directory inference_grap
    
#testing
-edit the TF-image-od.py file 
-change the testing image location 
-edit threshold (0.5) higher means higher accuracy

