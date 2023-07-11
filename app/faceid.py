# Import kivy dependencies 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#Build app and layout
class CamApp(App):

    def build(self):
        #Main Layout components 
        self.webcam=Image(size_hint=(1,.8))
        self.button=Button(text="Verify",on_press=self.verify,size_hint=(1,.1))
        self.verification_label=Label(text='Verification Uninitiated',size_hint=(1,.1))
        
        #Add items to layout
        layout=BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #Load model
        self.model=tf.keras.models.load_model('siamese_model.h5',custom_objects={'L1Dist':L1Dist})

        #Setup video capture device
        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)

        return layout
    
    #Run continuously to get webcam feed
    def update(self,*args):
        
        #Read frame from opencv
        ret,frame=self.capture.read()
        frame=frame[120:120+250,200:200+250, :]

        #Flip horizontal and convert image to texture
        buf=cv2.flip(frame,0).tostring()
        #Converts image to texture(allows image to display on screen in kivy)
        img_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.webcam.texture=img_texture

    #Load image and convert to 100 x 100 pixels
    def preprocess(self,file_path):
        #Read in image from file path
        byte_img=tf.io.read_file(file_path)
        #Load in the image
        img=tf.io.decode_jpeg(byte_img)
        img=tf.image.resize(img,(100,100))
        img=img/255.0
        return img
    
    def verify(self,*args):
        '''
        Detection Threhold-Metric above which image is considered positive
        Verification Threhold-Proportion of positive predictions/total positive samples
        '''
        #Specify Threshold
        detection_threshold=0.5
        verification_threshold=0.8

        #Capture input image from device
        SAVE_PATH=os.path.join('application_data','input_images','input_image.jpg')
        ret,frame=self.capture.read()
        frame=frame[120:120+250,200:200+250, :]
        cv2.imwrite(SAVE_PATH,frame)

        #build results array
        results=[]
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img=self.preprocess(os.path.join('application_data','input_images','input_image.jpg'))
            validation_img=self.preprocess(os.path.join('application_data','verification_images',image))

            result=self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            results.append(result)

        detection=np.sum(np.array(results)>detection_threshold)
        verification=detection/len(os.listdir(os.path.join('application_data','verification_images')))
        verified=verification>verification_threshold

        #Set verification text
        self.verification_label.text='Verified' if verified==True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(verification)


        return verified
    
if __name__=='__main__':
    CamApp().run()