class FlaskAppSetup:
    def __init__(self):

        # Initializes a Flask application.
        self.app = Flask(__name__)

        '''
        Initializes a boolean variable app_running and sets it to True. 
        This variable is used later to control the running state of the Flask application.
        '''

        self.app_running = True  