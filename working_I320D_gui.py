import tkinter as tk
import xgboost as xgb
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the model
with open('xgb_clf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the scaler object
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


print("Model loaded successfully.")
print("Preprocessor loaded successfully.")
print("Scaler loaded successfully.")


def get_inputs():
    loan_amnt = float(loan_amnt_entry.get())
    term = float(term_entry.get())
    int_rate = float(int_rate_entry.get())
    emp_length = float(emp_length_entry.get())
    home_ownership = home_ownership_var.get()
    annual_inc = float(annual_inc_entry.get())
    verification_status = verification_status_var.get()
    purpose = purpose_var.get()
    dti = float(dti_entry.get())
    inq_last_6mths = float(inq_last_6mths_entry.get())
    open_acc = float(open_acc_entry.get())
    revol_bal = float(revol_bal_entry.get())
    revol_util = float(revol_util_entry.get())
    initial_list_status = initial_list_status_var.get()
    total_rec_int = float(total_rec_int_entry.get())
    last_week_pay = float(last_week_pay_entry.get())
    tot_cur_bal = float(tot_cur_bal_entry.get())
    grade = grade_var.get()

    input_data = [[loan_amnt, term, int_rate, emp_length, home_ownership, annual_inc, verification_status, purpose, dti, inq_last_6mths, open_acc, revol_bal, revol_util, initial_list_status, total_rec_int, last_week_pay, tot_cur_bal, grade]]
    input_df = pd.DataFrame(input_data, columns=['loan_amnt', 'term', 'int_rate', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'initial_list_status', 'total_rec_int', 'last_week_pay', 'tot_cur_bal', 'grade'])

    preprocessed_data = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_data)

    if prediction[0] == 1:
        result_label.config(text="Eligible for loan", font=("Arial", 16), fg="green")
    else:
        result_label.config(text="Not eligible for loan", font=("Arial", 16), fg="red")

# Create the GUI
root = tk.Tk()
root.title("Loan Eligibility Prediction")



# Create a canvas and a scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold the content
content_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Bind the MouseWheel event to the canvas
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

loan_amnt_label = tk.Label(content_frame, text="Loan Amount")
loan_amnt_label.pack()
loan_amnt_entry = tk.Entry(content_frame)
loan_amnt_entry.pack()

term_label = tk.Label(content_frame, text="Term")
term_label.pack()
term_entry = tk.Entry(content_frame)
term_entry.pack()

int_rate_label = tk.Label(content_frame, text="Interest Rate")
int_rate_label.pack()
int_rate_entry = tk.Entry(content_frame)
int_rate_entry.pack()

emp_length_label = tk.Label(content_frame, text="Employment Length")
emp_length_label.pack()
emp_length_entry = tk.Entry(content_frame)
emp_length_entry.pack()

home_ownership_label = tk.Label(content_frame, text="Home Ownership")
home_ownership_label.pack()
home_ownership_var = tk.StringVar(value="ANY")
home_ownership_options = ["ANY", "MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]
for option in home_ownership_options:
    tk.Radiobutton(content_frame, text=option, variable=home_ownership_var, value=option).pack()

annual_inc_label = tk.Label(content_frame, text="Annual Income")
annual_inc_label.pack()
annual_inc_entry = tk.Entry(content_frame)
annual_inc_entry.pack()

verification_status_label = tk.Label(content_frame, text="Verification Status")
verification_status_label.pack()
verification_status_var = tk.StringVar(value="Not Verified")
verification_status_options = ["Not Verified", "Source Verified", "Verified"]
for option in verification_status_options:
    tk.Radiobutton(content_frame, text=option, variable=verification_status_var, value=option).pack()

# Add labels and entry boxes for each input variable
grade_label = tk.Label(content_frame, text="Grade")
grade_label.pack()
grade_var = tk.StringVar(value="G")
grade_options = ["A", "B", "C", "D", "E", "F", "G"]
for option in grade_options:
    tk.Radiobutton(content_frame, text=option, variable=grade_var, value=option).pack()


purpose_label = tk.Label(content_frame, text="Loan Purpose")
purpose_label.pack()
purpose_var = tk.StringVar(value="car")
purpose_options = ["car", "credit_card", "debt_consolidation", "educational", "home_improvement", "house", "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"]
purpose_dropdown = tk.OptionMenu(content_frame, purpose_var, *purpose_options)
purpose_dropdown.pack()

dti_label = tk.Label(content_frame, text="Debt-to-Income Ratio")
dti_label.pack()
dti_entry = tk.Entry(content_frame)
dti_entry.pack()

inq_last_6mths_label = tk.Label(content_frame, text="Inquiries in Last 6 Months")
inq_last_6mths_label.pack()
inq_last_6mths_entry = tk.Entry(content_frame)
inq_last_6mths_entry.pack()

open_acc_label = tk.Label(content_frame, text="Open Accounts")
open_acc_label.pack()
open_acc_entry = tk.Entry(content_frame)
open_acc_entry.pack()

revol_bal_label = tk.Label(content_frame, text="Revolving Balance")
revol_bal_label.pack()
revol_bal_entry = tk.Entry(content_frame)
revol_bal_entry.pack()

revol_util_label = tk.Label(content_frame, text="Revolving Utilization")
revol_util_label.pack()
revol_util_entry = tk.Entry(content_frame)
revol_util_entry.pack()

initial_list_status_label = tk.Label(content_frame, text="Initial List Status")
initial_list_status_label.pack()
initial_list_status_var = tk.StringVar(value="f")
initial_list_status_options = ["f", "w"]
for option in initial_list_status_options:
    tk.Radiobutton(content_frame, text=option, variable=initial_list_status_var, value=option).pack()

total_rec_int_label = tk.Label(content_frame, text="Total Interest Received")
total_rec_int_label.pack()
total_rec_int_entry = tk.Entry(content_frame)
total_rec_int_entry.pack()

last_week_pay_label = tk.Label(content_frame, text="Last Week Pay")
last_week_pay_label.pack()
last_week_pay_entry = tk.Entry(content_frame)
last_week_pay_entry.pack()

tot_cur_bal_label = tk.Label(content_frame, text="Total Current Balance")
tot_cur_bal_label.pack()
tot_cur_bal_entry = tk.Entry(content_frame)
tot_cur_bal_entry.pack()

# Add a button to submit the inputs and make a prediction
submit_button = tk.Button(content_frame, text="Submit", command=get_inputs)
submit_button.pack()

# Add a label to display the prediction result
result_label = tk.Label(content_frame, text="")
result_label.pack()

# Update the content frame size
content_frame.update_idletasks()
canvas.configure(scrollregion=canvas.bbox("all"))

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Set the window size
root.geometry("150x500")

# Run the Tkinter event loop
root.mainloop()

