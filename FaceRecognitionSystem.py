
class FaceRecognitionSystem(FlaskAppSetup, ImageProcessing, FolderFileSetup,AttendanceManager,WebInterface):
    
    # Current date formatted as "dd-Month-YYYY".
    datetoday2 = date.today().strftime("%d-%B-%Y")
    def __init__(self): 
        FlaskAppSetup.__init__(self)
        ImageProcessing.__init__(self)
        FolderFileSetup.__init__(self)
        AttendanceManager.__init__(self)
        self.setup_folders()
        self.setup_files()
        self.setup_routes()

        
        
        # Current date formatted as "dd-Month-YYYY".
        datetoday2 = date.today().strftime("%d-%B-%Y")
        # def __init__(self, face_recognition_system,AttendanceManager)
        self.web_interface = WebInterface(self,AttendanceManager)
        self.attendance_manager = AttendanceManager()  # Initialize the AttendanceManager instance
    
    # -----------------
    def run(self):
        self.app.run()    

    # Polymorphism: Same method name across different classes with different implementations
    def setup_routes(self):
        WebInterface.__init__(self, FaceRecognitionSystem,AttendanceManager)
        print("Setting up Flask routes for face recognition system.")

    def detect_faces(self, image):
        print("Detecting faces using OpenCV.")

    def setup_folders(self):
        FolderFileSetup.__init__(self)
        print("Setting up folders for face recognition system.")

    def setup_files(self):
        print("Setting up files for face recognition system.")

