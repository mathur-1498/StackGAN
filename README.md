READ ME: AN IMPLEMENTATION OF GENERATIVE ADVERSARIAL NETWORKS

Since the implantation/ execution of code requires processing of thousands of images, it requires high usage of GPU. So we use Google Colab PRO to run this code.

Dependencies

python 3.7
TensorFlow 2
Torch is needed, to use the pre-trained DAMSM text encoder.
In addition, please add the project folder to PYTHONPATH and pip install the following packages:
•	prettytensor
•	Numpy
•	opencv
•	pickle
•	pandas
•	torchfile

Data

1.	Download author preprocessed DAMSM text embeddings for birds and save them to Data/.
2.	Download the birds image data. Extract them to Data/birds/.
3.	Preprocess images.
•	For birds: python code/preprocess_birds.py

Training

1.	Execute the data preprocessing code
2.	Execute the stack 1 GAN
3.	Execute the stack 2 GAN
4.	Save the weights
5.	Function to generate image based on user input caption

Execution STEPS:

1.	Upload the code folder which has the already pre-trained model that we did to google drive.
2.	Open google colab to run (make sure you have the PRO version)
3.	Mount Google drive
4.	Access working directory through terminal ( inside the code folder)
5.	Execute app.py (this is the UI web app through which we’ll be running)
