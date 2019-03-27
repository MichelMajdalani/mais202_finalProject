from flask import Flask, render_template, request, url_for, jsonify, send_file, redirect, send_from_directory
from werkzeug.utils import secure_filename
from predictor import Predictor
from state import StateTracker

UPLOAD_FOLDER = 'model/'
ALLOWED_EXTENSIONS = set(['csv'])
# initialize flask app
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# initialize the predictor object
predictor = Predictor()

# intialize the state tracker object
state_tracker = StateTracker()

# routing functionality for root url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # update the current state
        state_tracker.updateState(request)
        # make prediction using model
        pred = predictor.predict(request)
        # update the result in current state
        state_tracker.state['result'] = 'Malignant' if pred[0] == 'M' else 'Benign'
        # rerender template with updated states
        return render_template('index.html', state=state_tracker.state)
    else:
        return render_template('index.html', state=state_tracker.state)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return


@app.route('/')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
