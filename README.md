# diabetic-retinopathy-detection
Diabetic Retinopathy Detection using Deep Learning
This project was developed as a graduation project by Rabia Biçer and Ayşe Şevval Yılmaz.
The aim of this study is early detection of diabetic retinopathy from retinal fundus images using deep learning.
## Project Structure
notebooks/
src/
images/
requirements.txt
## Trained Model
The trained model file is large and cannot be hosted directly on GitHub.
You can download it here:
[Download model.h5](https://drive.google.com/file/d/1BBDgM4Du11GIdRpi6E49P1TjxSems8Ed/view?usp=sharing)


## ▶️ How to Run

Follow these steps to run the project locally.

### 1) Download the repository
Click the green **Code** button and select **Download ZIP**, then extract the folder.

### 2) Download the trained model
Download `model.h5` from the link provided in this README and place it in the project root directory (same folder as `app.py`).

The folder structure should look like this:
```
project_folder/
│── app.py
│── requirements.txt
│── model.h5
```

### 3) Open terminal in the project folder
Open the project folder and launch a terminal inside it.

### 4) Install dependencies
Run the following command:

```
pip install -r requirements.txt
```

### 5) Run the Streamlit app
Run:

```
streamlit run app.py
```

### 6) Open the app in browser
After running the command, Streamlit will provide a local URL.  
Open the link in your browser and upload a fundus image to get predictions.
