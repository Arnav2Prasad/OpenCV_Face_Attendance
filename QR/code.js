/*
This code is written in Google Apps Script, a JavaScript-based scripting language that allows you to
automate tasks and interact with various Google services such as Google Sheets, 
Google Drive, and Google Forms.

this script is designed to be used in conjunction with a Google Sheets
 form. When called, it saves latitude, longitude, and location pin data
  into specific columns of the Google Sheet named "Form Responses 7".

  The main purpose of this code is to get the exact user location marking his/her attendance
  so that the authorities can get to know whether he is really present in the lecture or seminars, or
  marking false attendance(issue pretains in current attendance system)

  This system provides authorities with a reliable method to confirm attendance, 
  reducing manual errors and enhancing security.
*/



/**
 * function that gets invoked when a user accesses the web app URL. 
 * It returns an HTML file named 'index' using HtmlService.createHtmlOutputFromFile.  
 
 * 'e' contains --> event object that contains information about the HTTP request 
 * sent to the web app 
 * @param {*} e 
 * @returns 
 */
function doGet(e) {
    return HtmlService.createHtmlOutputFromFile('index');
}

/**
 * This function takes three parameters(or arguments): latitude, longitude, and locationpin. 
 * It's intended to be called to save coordinates and location pins to a Google Sheet and so 
 * get into Google Forms.
 * @param {*} latitude 
 * @param {*} longitude 
 * @param {*} locationpin 
 */
function saveCoordinates(latitude,longitude, locationpin){
    
    //Declared two variables latlongcol and pinlocationcol to store the column numbers for latitude
    //and longitude, and location pin respectively.
    
    var latlongcol, pinlocationcol;

    // to link with google spreadsheet named : Form Responses 7
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Form Responses 7");

    //to get into last row of spreadsheet
    var lastrow= sheet.getLastRow(); 

    //to get header of spreadsheet
    var header = sheet.getRange(1, 1, 1,sheet.getLastColumn()).getValues();
    
    // Flattens the header row array to a single-dimensional array.
    header=header.flat();  

    
    //Loops through the header to find the columns for latitude and longitude, and location pin.
    //If the column names "LatLong" and "GeoLocation" are found in the header, it stores 
    //their respective column numbers.
    for(var i=1;i<=header.length;i++){
      
      if(header[i]=="LatLong")
      {
        latlongcol=i+1;
      }
      
      if(header[i]=="GeoLocation"){
      
        pinlocationcol=i+1;
      }
    
    }
    //Checks if the last row has data, and if the latitude, longitude, and location pin columns are empty.
    if(sheet.getRange(lastrow, 1).getValue()!="" && sheet.getRange(lastrow, latlongcol).getValue()==""  &&  sheet.getRange(lastrow, pinlocationcol).getValue()==""){
        
        //Sets the latitude and longitude values in the appropriate column.
        sheet.getRange(lastrow, latlongcol).setValue(latitude +" , "+ longitude);

        //Sets the location pin value in the appropriate column.
        sheet.getRange(lastrow, pinlocationcol).setValue(locationpin);
    }
  }
