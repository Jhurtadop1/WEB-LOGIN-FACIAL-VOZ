
import zlib
from werkzeug.utils import secure_filename
from flask import Response
import cv2
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
import re
from helpers import apology, login_required
from flask import Flask, render_template, request, url_for, redirect, session
import pymongo
import bcrypt
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_slim
import os
from functools import wraps






# Configure application
app = Flask(__name__)
# configure flask-socketio

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


client = pymongo.MongoClient(
    "mongodb+srv://hugo:hugo@cluster0.zdsz6w8.mongodb.net/")
db = client.get_database('login')
users = db.users

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
@login_required
def home():
    return redirect("/home")


@app.route("/home")
@login_required
def index():
    if "user_id" in session:
        username = session["user_id"]
        return render_template("index.html", username=username)
    else:
        return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html", messager=1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("login.html", messager=2)

        # Query database for username
        name_found = users.find_one({"name": input_username})
        if name_found:
            name_val = name_found['name']
            passwordcheck = name_found['password']

            if bcrypt.checkpw(input_password.encode('utf-8'), passwordcheck):
                # Remember which user has logged in
                session["user_id"] = name_val
                # Redirect user to home page
                return redirect(url_for('index'))

            else:
                return render_template("login.html", messager=3)
        else:
            return render_template("login.html", messager=4)  # Mensaje de usuario no encontrado

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")
@app.route("/success")
def success():

    return render_template("success.html")
@app.route("/logout")
def logout():
    """Log user out"""
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html", messager=1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html", messager=2)

        # Ensure password confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html", messager=4)

        # Check if passwords match
        elif not input_password == input_confirmation:
            return render_template("register.html", messager=3)

        # Query database for username
        user_found = users.find_one({"name": input_username})
        if user_found:
            return render_template("register.html", messager=5)

        # Insert new user into the database
        else:
            hashed = bcrypt.hashpw(input_password.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': input_username, 'password': hashed}
            users.insert_one(user_input)

            new_user = users.find_one({'name': input_username})
            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user["name"]

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("index.html")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")
    
@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    similarity_percentage = None  # Inicializa la variable para almacenar el porcentaje de similitud
    
    if request.method == "POST":

        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        username = request.form.get("name")

        id_ = username
        compressed_data = zlib.compress(encoded_image, 5)

        uncompressed_data = zlib.decompress(compressed_data)

        decoded_data = b64decode(uncompressed_data)

        new_image_handle = open('./static/validarface/' + str(id_) + '.jpg', 'wb')

        new_image_handle.write(decoded_data)
        new_image_handle.close()
        try:
            image_of_user = face_recognition.load_image_file(
                './static/face/' + str(id_) + '.jpg')
        except:
            return render_template("camera.html", message=5)

        image_of_user = cv2.cvtColor(image_of_user, cv2.COLOR_BGR2RGB)
        user_face_encoding = face_recognition.face_encodings(image_of_user)[0]

        unknown_image = face_recognition.load_image_file(
            './static/validarface/' + str(id_) + '.jpg')
        try:

            unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except:
            return render_template("camera.html", message=2)

        #  compare faces
        results = face_recognition.compare_faces(
            [user_face_encoding], unknown_face_encoding)
         # Calcula el porcentaje de similitud
        similar_faces_count = sum(results)
        total_faces = len(results)
        similarity_percentage = (similar_faces_count / total_faces) * 80

        # Determina un umbral para mostrar un mensaje
        threshold = 50  # Puedes ajustar este valor según tu preferencia

        if similarity_percentage >= threshold:
            similarity_message = "Las caras son similares"
        else:
            similarity_message = "Las caras no son lo suficientemente similares"

        if results[0]:
            user_found = users.find_one({"name": id_})
            session["user_id"] = user_found["name"]
            return render_template("success.html", username=user_found["name"], similarity_message=similarity_message, similarity_percentage=similarity_percentage)
        else:
            return render_template("camera.html", message=3, similarity_message=similarity_message)

    else:
        return render_template("camera.html", similarity_percentage=similarity_percentage)


@app.route("/facesetup", methods=["GET", "POST"])
@login_required
def facesetup():
    if request.method == "POST":

        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        user_name=session["user_id"]
        user_found = users.find_one({"name":user_name })
        id_ = user_found["name"]

        # id_ = db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]
        compressed_data = zlib.compress(encoded_image, 5)

        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)

        new_image_handle = open('./static/face/' + str(id_) + '.jpg', 'wb')

        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_user = face_recognition.load_image_file(
            './static/face/' + str(id_) + '.jpg')
        try:
            user_face_encoding = face_recognition.face_encodings(image_of_user)[0]
        except:
            return render_template("face.html", message=1)
        return redirect("/home")

    else:
        return render_template("face.html")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html", e=e)


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
#--------------------------------------------------------------------------------------------------------------------------# implementación voz

# Configuración para la carga de archivos
UPLOAD_FOLDER = 'static/voice'
VALIDAR_UPLOAD_FOLDER = 'static/validarvoice'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VALIDAR_UPLOAD_FOLDER'] = VALIDAR_UPLOAD_FOLDER

# Cargar el modelo VGGish
try:
    tf.compat.v1.disable_eager_execution()
    ckpt_path = "vggish_model.ckpt"
    sess = tf.compat.v1.Session()
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, ckpt_path)
except Exception as e:
    print(f"Error al cargar el modelo VGGish: {e}")

# Función para extraer características de audio
def extract_audio_features(audio_path):
    try:
        features = vggish_input.wavfile_to_examples(audio_path)
        features_embed = sess.run('vggish/embedding:0', feed_dict={'vggish/input_features:0': features})
        return features_embed.ravel()
    except Exception as e:
        print(f"Error al extraer características de audio: {e}")
        return None

# Función para comparar características de audio
def compare_audio_features(features1, features2):
    try:
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return similarity
    except Exception as e:
        print(f"Error al comparar características de audio: {e}")
        return 0.0

# Función para convertir el archivo de audio al formato WAV
def convert_to_wav(audio_file, output_path):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error al convertir archivo a WAV: {e}")
        return False
    
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" in session:
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrapper


@app.route('/registrarvoz')
@login_required
def registrarvoz():
    return render_template('RegistrarVoz.html')

@app.route('/uploadRegistrar', methods=['POST'])
@login_required
def upload_registrar():
    # Obtener el nombre de usuario de la sesión actual
    username = session.get('user_id')

    # Validar si hay un usuario en sesión
    if username:
        # Proceder con la lógica de grabación y almacenamiento de audio
        audio_file = request.files['audio']

        # Convertir el archivo de audio al formato WAV antes de guardarlo
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{username}.wav')
        convert_to_wav(audio_file, audio_path)

        return jsonify({"message": f"Voz registrada correctamente para: {username}"})
    else:
        return jsonify({"message": "No hay un usuario en sesión. Por favor, inicie sesión para registrar su voz."})

@app.route('/checkUserExistence')
def check_user_existence():
    # Verificar la existencia del usuario en la base de datos
    username = request.args.get('username')
    user_exists = bool(db.users.find_one({"name": username}))
    return jsonify({"exists": user_exists})

@app.route('/validarvoz')
def validarvoz():
    return render_template('ValidarVoz.html')

@app.route('/compareAndStoreValidarVoice', methods=['POST'])
def compare_and_store_validar_voice():
    # Obtener el nombre de usuario desde el formulario
    username = request.form.get('username')

    # Validar si el nombre de usuario existe en la base de datos
    if db.users.find_one({"name": username}):
        # El usuario existe, proceder con la lógica de comparación y almacenamiento en validarvoice
        audio_file = request.files['audio']

        # Guardar el archivo de audio en la carpeta 'static/validarvoice' con el nombre del usuario
        audio_path_validar = os.path.join(app.config['VALIDAR_UPLOAD_FOLDER'], f'{username}.wav')
        convert_to_wav(audio_file, audio_path_validar)

        # Extraer características de audio de ambas grabaciones
        features_embed_voice1 = extract_audio_features(audio_path_validar)
        features_embed_voice2 = extract_audio_features(os.path.join(app.config['UPLOAD_FOLDER'], f'{username}.wav'))

        if features_embed_voice1 is not None and features_embed_voice2 is not None:
            # Comparar características de audio
            similarity = compare_audio_features(features_embed_voice1, features_embed_voice2)

            # Determinar si las voces coinciden o no
        if similarity > 0.7:  # Ajustar el umbral según sea necesario
            session["user_id"] = username  # Establecer la sesión para el usuario
        return redirect(url_for('success'))  # Redirigir al usuario a success.html
    else:
        return jsonify({"message": "La voz no coincide."})

if __name__ == '__main__':
    app.run()
