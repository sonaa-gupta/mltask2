#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import vgg16


# In[2]:


from keras.layers import  Dense 


# In[3]:


model = vgg16.VGG16(weights= 'imagenet' , include_top = False ,input_shape= (224,224,3))





# In[4]:


m=0
for i in model.layers:
    m = m+1
print(m)


# In[5]:


model.layers


# In[ ]:





# In[6]:


model.layers[0].trainable = False


# In[7]:


model.layers[0]


# In[8]:


model.layers[0].trainable


# In[9]:


model.layers[1].trainable


# In[10]:


for i in model.layers:
    i.trainable = False
for (j,layer) in enumerate (model.layers):
     print(str(i)+" " + layer.__class__.__name__,layer.trainable)


# In[11]:


model.output


# In[12]:


model.summary()


# In[13]:


from keras.layers import GlobalAveragePooling2D,Conv2D


# In[14]:


top_model = model.output
#top_model = Conv2D()
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(1024,activation = 'relu')(top_model)


# In[ ]:





# In[15]:


top_model = Dense(1024 , activation = 'relu')(top_model)
top_model = Dense(512 , activation = 'relu')(top_model)
top_model  = Dense(256 , activation = 'relu')(top_model)
top_model = Dense(3,activation = 'softmax')(top_model)


# 

# In[16]:


from keras.models import Sequential ,Model
from keras.layers import Flatten,MaxPooling2D ,Conv2D


# In[17]:


model = Model(inputs = model.input , outputs = top_model)


# In[18]:


print(model.summary())


# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# train_datagen = ImageDataGenerator(
#                                  rescale = 1./255,
#                                  rotation_range = 45,
#                                  width_shift_range = 0.3,
#                                  height_shift_range = 0.3,
#                                  horizontal_flip = True,
#                                  fill_mode = 'nearest')
# validation_datgen = ImageDataGenerator(rescale = 1./255)
# batch_size = 32
# train_generator = train_datgen.flow_from_directory(
#                    'facetrain'
#                     target_size = (224,224,3)
#                     batch_size = batch_size
#                     class_mode = 'categorical')
# validation_generator

# In[20]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 45,
                                   width_shift_range = 0.3,
                                   zoom_range=0.2,
                                   shear_range = 0.2,
                                   height_shift_range = 0.3,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest') 
validation_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 32 
train_generator = train_datagen.flow_from_directory('mldata/facetrain',
                                                   target_size = (224,224),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory('/mldata/New folder (2)',
                                                             target_size = (224,224),
                                                             batch_size = batch_size,
                                                             class_mode = 'categorical')


# In[21]:



from keras.callbacks import ModelCheckpoint , EarlyStopping


# In[22]:


Checkpoint =ModelCheckpoint('face_recognition' ,
                            monitor ='val_loss',
                            verbose=1,
                            mode = 'min',
                            save_best_only=True)
Stopping = EarlyStopping( monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True,)
callback = [Checkpoint , Stopping]
model.compile(optimizer = 'adam' , loss ='categorical_crossentropy' , metrics=['accuracy'])
epoch = 3
batch_size = 16
history = model.fit_generator(train_generator,
                             steps_per_epoch = 337//batch_size, 
                             epochs= epoch,
                             callbacks = callback,
                             validation_data = validation_generator,
                             validation_steps = 17//batch_size)
                    


# In[23]:





# In[ ]:





# In[ ]:





# In[ ]:




