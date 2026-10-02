🚢 Titanic Survival Prediction
Logistic Regression Machine Learning Project
An interactive Titanic Survival Prediction web application built using 
Python,
 Machine Learning, 
NumPy, 
Pandas, 
Streamlit.
The application uses a trained Logistic Regression model to predict whether a Titanic passenger would survive based on passenger information such as passenger class, sex, age, family information, fare, and embarkation port.

🌐 Live Project
🚀 Streamlit App:
https://titanic-survival-prediction-project-btamxmxaje58znpqgmwnmf.streamlit.app/


👨‍💻 Developer
Name: Yanaguntikar Meesal

📧 Email: yanaguntikarm@gmail.com

📌 Project Overview
The Titanic Survival Prediction project is a Machine Learning classification project based on the famous Titanic dataset.
The project uses Logistic Regression to predict the survival outcome of a passenger.
The user enters passenger details through an interactive Streamlit interface, and the trained machine learning model generates a prediction.
Prediction Classes
Value
Meaning
0
❌ Did Not Survive
1
✅ Survived


🎯 Project Objectives
The main objectives of this project are:
Understand binary classification using Machine Learning.
Implement Logistic Regression.
Prepare Titanic passenger data for Machine Learning.
Train a classification model.
Save the trained model using Pickle.
Build an interactive Streamlit application.
Accept passenger information from users.
Generate survival predictions.
Display prediction results in an easy-to-understand format.
Deploy the Machine Learning application as a web application.

🛠️ Technologies Used
Technology
Purpose
🐍 Python
Programming language
🤖 Logistic Regression
Machine Learning algorithm
📊 Pandas
Data handling and analysis
🔢 NumPy
Numerical operations
🎨 Streamlit
Web application
📦 Pickle
Model loading
💻 GitHub
Project version control
☁️ Streamlit Cloud
Application deployment


📊 Machine Learning Algorithm
Logistic Regression
Logistic Regression is a supervised Machine Learning algorithm commonly used for binary classification problems.
In this project, the model predicts two possible outcomes:
0 → Did Not Survive

1 → Survived

The model learns relationships between passenger features and the historical survival outcome.

📥 Input Features
The Streamlit application accepts the following passenger information:
1. Pclass
Passenger class:
1 → First Class
2 → Second Class
3 → Third Class

2. Sex
Encoded as:
0 → Female
1 → Male

3. Age
Passenger age.
4. SibSp
Number of siblings or spouses aboard the Titanic.
5. Parch
Number of parents or children aboard the Titanic.
6. Fare
Passenger ticket fare.
7. Embarked
Port of embarkation:
0 → Cherbourg
1 → Queenstown
2 → Southampton

These numerical encodings must match the encoding used when the Machine Learning model was trained.

🧠 Model Input
The application creates the model input in this order:
[
    Pclass,
    Sex,
    Age,
    SibSp,
    Parch,
    Fare,
    Embarked
]

Example:
Pclass     = 1
Sex        = 0
Age        = 30
SibSp      = 0
Parch      = 0
Fare       = 50
Embarked   = 2

The input is converted into a NumPy array:
input_data = np.array(
    [[
        Pclass,
        Sex,
        Age,
        SibSp,
        Parch,
        Fare,
        Embarked
    ]]
)

The trained model then generates the prediction:
prediction = model.predict(input_data)


🔄 Project Workflow
               Titanic Dataset
                       ↓
                Data Cleaning
                       ↓
                Data Preparation
                       ↓
              Feature Selection
                       ↓
             Feature Encoding
                       ↓
             Train/Test Split
                       ↓
             Logistic Regression
                       ↓
               Model Training
                       ↓
             Save Model (.pkl)
                       ↓
              Streamlit Application
                       ↓
             User Passenger Input
                       ↓
              Model Prediction
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        Prediction = 0     Prediction = 1
              ↓                 ↓
       ❌ Not Survived      ✅ Survived


🖥️ Application Features
🚢 Titanic Survival Prediction
The application provides a simple interface for entering passenger information.

👤 Passenger Details
Users can enter:
Passenger Class
Gender
Age
Number of Siblings/Spouses
Number of Parents/Children
Fare
Embarkation Port

🔮 Prediction Button
After entering the passenger information, users can click:
🔮 Predict Survival

The application sends the information to the trained Machine Learning model.

✅ Prediction Result
If the model predicts:
1

the application displays:
✅ Result: Survived

If the model predicts:
0

the application displays:
🚩 Result: Not Survived


📁 Project Structure
Titanic-Survival-Prediction/
│
├── app.py
│
├── mode_titanic.pkl
│
├── README.md
│
└── requirements.txt

app.py
Contains the Streamlit application and prediction logic.
mode_titanic.pkl
Contains the trained Machine Learning model saved using Pickle.
requirements.txt
Contains the Python libraries required to run the application.
README.md
Contains the complete project documentation.

📦 Requirements
Create a file named:
requirements.txt

Add:
streamlit
numpy
pandas

If your model was trained using scikit-learn and the Pickle file depends on it, also include:
scikit-learn

Therefore, the recommended requirements.txt is:
streamlit
numpy
pandas
scikit-learn


💻 Installation
Step 1: Clone the Repository
git clone https://github.com/your-username/Titanic-Survival-Prediction.git

Move into the project directory:
cd Titanic-Survival-Prediction


Step 2: Create a Virtual Environment
Windows
python -m venv .venv

Activate it:
.venv\Scripts\activate


Step 3: Install Dependencies
pip install -r requirements.txt

Or install the libraries directly:
pip install streamlit numpy pandas scikit-learn


▶️ Run the Application
Start Streamlit:
streamlit run app.py

If that command does not work:
python -m streamlit run app.py

The application will open in your browser.

⚠️ Common Error
Model File Not Found
If you see:
Model file not found!

make sure the model file is located in the same directory as app.py.
Correct structure:
Titanic-Survival-Prediction/
│
├── app.py
├── mode_titanic.pkl
├── requirements.txt
└── README.md


🔧 Important Model Filename
The application currently loads:
pickle.load(open("mode_titanic.pkl", "rb"))

Therefore, the file must be named exactly:
mode_titanic.pkl

If your actual model file is named:
model_titanic.pkl

change the code to:
model = pickle.load(
    open("model_titanic.pkl", "rb")
)

The filename in app.py and the actual .pkl file must match.

📈 Example Prediction
Suppose the user enters:
Pclass       = 1
Sex          = Female
Age          = 30
SibSp        = 0
Parch        = 0
Fare         = 50
Embarked     = Southampton

The application converts these values into numerical features and sends them to the trained Logistic Regression model.
The model returns either:
0

or:
1

The Streamlit application then displays the corresponding result.

🎓 Machine Learning Concepts Demonstrated
This project demonstrates practical knowledge of:
Supervised Learning
Binary Classification
Logistic Regression
Feature Selection
Feature Encoding
Training and Testing
Model Prediction
Model Serialization
Pickle
NumPy
Pandas
Streamlit
Machine Learning Deployment

💼 Resume Project Description
Titanic Survival Prediction | Python, Machine Learning, Streamlit
Developed an interactive Titanic Survival Prediction application using Python and Logistic Regression. Built a Streamlit web interface that accepts passenger class, gender, age, family information, fare, and embarkation details to generate survival predictions. Serialized the trained Machine Learning model using Pickle and integrated it into the Streamlit application for real-time prediction.

🎤 Interview Explanation
If an interviewer asks:
"Explain your Titanic project."
You can say:
I developed a Titanic Survival Prediction project using Python and Logistic Regression. The objective was to build a binary classification model that predicts whether a passenger survived based on passenger information. I used features such as passenger class, sex, age, number of siblings or spouses, parents or children, fare, and embarkation port. After training the model, I saved it using Pickle and developed a Streamlit application to provide an interactive user interface. The user enters passenger details, and the application passes the information to the trained model and displays whether the predicted outcome is survived or not survived.

🚀 Future Improvements
The project can be improved by adding:
📊 Prediction probability
📈 Model accuracy
📉 Confusion matrix
📊 Classification report
📈 Feature importance
👥 Passenger dataset exploration
📊 Interactive Titanic charts
📥 Prediction history
📥 Download prediction results
🎨 Advanced Streamlit UI
☁️ Streamlit Cloud deployment
🔍 Batch prediction using CSV files

🌐 Deployment
The application can be deployed using Streamlit Community Cloud.
Basic deployment process:
GitHub Repository
       ↓
Upload Project Files
       ↓
Connect Repository to Streamlit
       ↓
Select app.py
       ↓
Deploy
       ↓
Live Streamlit Application


📌 Project Highlights
🚢 Titanic Survival Prediction
        ↓
🤖 Logistic Regression
        ↓
📊 Passenger Features
        ↓
🧮 Numerical Prediction
        ↓
🎨 Streamlit Interface
        ↓
✅ Survival Result


⭐ Skills Demonstrated
Python
   ↓
NumPy
   ↓
Pandas
   ↓
Machine Learning
   ↓
Logistic Regression
   ↓
Model Serialization
   ↓
Pickle
   ↓
Streamlit
   ↓
Web Application


👨‍💻 Author
Yanaguntikar Meesal
Data Science & Machine Learning Project
📧 Email: yanaguntikarm@gmail.com

📬 Contact
For questions, suggestions, or collaboration:
Yanaguntikar Meesal
📧 yanaguntikarm@gmail.com

⭐ Support
If you find this project useful for learning Python, Machine Learning, Logistic Regression, or Streamlit, you can ⭐ star the repository on GitHub.

🚢 Titanic Survival Prediction
Developed by Yanaguntikar Meesal
Python • Machine Learning • Logistic Regression • NumPy • Pandas • Streamlit

