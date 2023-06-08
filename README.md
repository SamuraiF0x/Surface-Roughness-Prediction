# Surface roughness prediction CNN

CNN for predicting surface roughness after milling machining process

## Starting out

ctrl + shift + P -> create env, choose interpreter (latest python version)

## Install libs

py -m pip install numpy

# First moderately successful model

Overfitting is possible, for although val_accuracy=1.00, the model sometimes fails to predict correctly on new images

[INFO] ... [127/127] images were preprocessed successfully
[INFO] Number of generated images: 1016
[INFO] Number of samples used for training ... 812
[INFO] Number of samples used for validation ... 204
25/25 [==============================] - 1364s 51s/step - loss: 1162.7634 - accuracy: 0.3573 - val_loss: 1.0552 - val_accuracy: 0.4706
[INFO] Current validation accuracy: 0.47058823704719543
Epoch 1/2
25/25 [==============================] - 656s 26s/step - loss: 1.1152 - accuracy: 0.4459 - val_loss: 0.9554 - val_accuracy: 0.5392
Epoch 2/2
25/25 [==============================] - 1276s 51s/step - loss: 0.9498 - accuracy: 0.5787 - val_loss: 0.7493 - val_accuracy: 0.8284
[INFO] Current validation accuracy: 0.8284313678741455
Epoch 1/3
25/25 [==============================] - 2660s 106s/step - loss: 0.5641 - accuracy: 0.8159 - val_loss: 0.3101 - val_accuracy: 0.9314
Epoch 2/3
25/25 [==============================] - 2721s 108s/step - loss: 0.1916 - accuracy: 0.9360 - val_loss: 0.0327 - val_accuracy: 1.0000
Epoch 3/3
25/25 [==============================] - 2640s 105s/step - loss: 0.0430 - accuracy: 0.9941 - val_loss: 0.0168 - val_accuracy: 1.0000
[INFO] Current validation accuracy: 1.0
[INFO] ... Saving surface roughness prediction model ...
[INFO] The surface roughness prediction model was built SUCCESSFULLY
[INFO] Run predictRoughness to test the model

# Second moderately successful model
