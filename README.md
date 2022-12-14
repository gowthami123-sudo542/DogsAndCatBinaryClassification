## BinaryClassification of Dogs and Cat

## Libraries
   * Tesorflow framework
   * keras
   * gradio

## Gradio

. Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!
## Steps to Follow:
  1) Download the dataset from Kaggle directly.(check the code)
     * open the kaggle account and go to your profile.
     * Next go to account and there you can find ***API*** section.
     * In API Section click the button which shows create a new API token(json file) will downloaded.

 2) Next,Go to Datasets Section and copy the API Command and paste in google colab which gives the dataset link.
 3) When you use the above command  you get the Zip folder so unzip the folder to get training and testing data separately.
 4) Import all necessary libraries.
 5) Use the generators function when you have uneven size of the images and also for large dataset.
 6) Normalize the data values to be between 0 to 255.
 7) Create a CNN model.
 8) Compile the model and train the model by using training dataset.
 9) By using ***Gradio*** library you can view the perofrmance of the model

## Hyper Parameters of the model
* Input Image Size=(256,256,3)
* Batch Size=32
* Kernel size=(3,3)
* padding='valid'
* Activation function='Relu' (for conv layers)
* Activation function='Sigmoid' (for classification)
* Max Pooling PoolSize=(2,2)
* stride=2
* optimizer='adam'
* loss='binary_crossentropy'
* metrics='accuracy'
* epochs=25

## Accuracy of the model
 * Accuracy:0.9951
 * val Accuracy:0.8256
 

## Results

![image](https://user-images.githubusercontent.com/67852967/197771065-25a476ab-2bba-4008-9ebf-4189584b4102.png)
![image](https://user-images.githubusercontent.com/67852967/197771224-4bc06827-24d2-4cc8-8421-e966fd324dcc.png)

## Try this in local machine

* download the .ipynb file and run locally to view results.
 
     
