from flask import Flask, render_template, request, jsonify
import os
import pandas as pd

app = Flask(__name__)

# Function to read Excel file and get attendance percentage for a roll number
def get_attendance_percentage(roll_number):
    file_path = os.path.join(app.static_folder, 'trial.xlsx')  # Update with your file path
    if not os.path.exists(file_path):
        return None  # File not found

    df = pd.read_excel(file_path)
    student = df[df['Roll No'] == roll_number]

    if student.empty:
        return None  # Roll number not found
    else:
        return student.iloc[0]['Attendance Percentage']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_attendance_percentage', methods=['POST'])
def get_attendance_percentage_route():
    roll_number = request.form.get('roll_number')
    attendance_percentage = get_attendance_percentage(roll_number)
    if attendance_percentage is not None:
        return jsonify({'attendance_percentage': attendance_percentage})
    else:
        return jsonify({'error': 'Roll number not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)

