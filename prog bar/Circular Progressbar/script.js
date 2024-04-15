// function setProgressEndValue() {
//   let rollNumber = parseInt(document.getElementById("rollNumberInput").value);
//   let fileInput = document.getElementById('fileInput');
//   let fileName = fileInput.files[0].name;

//   if (!isNaN(rollNumber)) {
//     getAttendancePercentage(rollNumber, fileName);
//   } else {
//     alert("Please enter a valid roll number.");
//   }
// }

// function getAttendancePercentage(rollNumber, fileName) {
//   const reader = new FileReader();
  
//   reader.onload = function (e) {
//     const data = new Uint8Array(e.target.result);
//     const workbook = XLSX.read(data, { type: 'array' });
//     const sheetName = workbook.SheetNames[0]; // Assuming the data is in the first sheet
//     const sheet = workbook.Sheets[sheetName];
    
//     // Convert the sheet to JSON object
//     const jsonData = XLSX.utils.sheet_to_json(sheet);
    
//     // Find the attendance percentage for the given roll number
//     const student = jsonData.find(item => item['Roll No'] === rollNumber);
    
//     if (student && student['Attendance Percentage']) {
//       progressEndValue = parseFloat(student['Attendance Percentage']);
//       updateProgress();
//     } else {
//       alert("Attendance percentage not found for the given roll number.");
//     }
//   };
  
//   reader.readAsArrayBuffer(fileName);
// }

let progressBar = document.querySelector(".circular-progress");
let valueContainer = document.querySelector(".value-container");

let progressValue = 0;
let progressEndValue = 0; // Initialize progressEndValue to 0 initially
let speed = 3;

let progress = null; // Initialize progress variable

function updateProgress() {
  clearInterval(progress);
  progress = setInterval(() => {
    progressValue++;
    valueContainer.textContent = `${progressValue}%`;
    progressBar.style.background = `conic-gradient(
        #4d5bf9 ${progressValue * 3.6}deg,
        #cadcff ${progressValue * 3.6}deg
    )`;
    if (progressValue == progressEndValue) {
      clearInterval(progress);
    }
  }, speed);
}

function setProgressEndValue() {
  let rollNumber = parseInt(document.getElementById("rollNumberInput").value);
  let xhr = new XMLHttpRequest();
  xhr.open('POST', '/get_attendance_percentage', true);
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      let response = JSON.parse(xhr.responseText);
      if ('attendance_percentage' in response) {
        progressEndValue = parseFloat(response.attendance_percentage);
        updateProgress(); // Update progress bar after setting new progressEndValue
      } else {
        alert(response.error);
      }
    }
  };
  xhr.send('roll_number=' + rollNumber);
}

