from utils import *

Custom_Model = Sequential()
Custom_Model.add(Conv2D(32, (3, 3), input_shape=(232, 232, 1), activation='relu'))
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Conv2D(64, (3, 3), activation='relu'))
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Conv2D(128, (3, 3), activation='relu'))
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Flatten())
Custom_Model.add(Dense(units=128, activation='relu'))
Custom_Model.add(Dense(units=3, activation='softmax'))  
Custom_Model.summary()

Custom_Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

fle_s=r'TestTraining/Custom_Model.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')

callback_list=[checkpointer]
history = Custom_Model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch
    ,epochs=epochs,shuffle=True,callbacks=[callback_list],
    validation_data=validation_generator,
    validation_steps=len(validation_generator.classes)
)


test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(232, 232),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

y_pred =Custom_Model.predict(test_generator)
y_true = test_generator.classes
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

Custom_Model.save('Custom_Model.h5')
print("Custom Model Training Completed!!!")