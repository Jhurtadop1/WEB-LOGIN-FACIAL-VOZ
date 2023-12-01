{% extends "layout.html" %}

{% block title %}
    Registrar Voz
{% endblock %}

{% block main %}
<!DOCTYPE html>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" integrity="sha512-z3gLpd7yknf1YoNbCzqRKc4qyor8gaKU1qmn+CShxbuBusANI9QpRohGBreCFkKxLhei6S9CQXFEbbKuqLg0DA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Recorder</title>
    <style>
        .alert {
            padding: 20px;
            background-color: #36f475;
            color: white;
            margin-bottom: 15px;
        }
        body {
            font-family: 'Arial', sans-serif;
            background-color: #222;
            color: white;
            text-align: center;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        #container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin: 0 50px; /* Added margin of 50px on the left and right */
            height: 100%;
        }

        h1 {
            color: #333;
        }

        label {
            display: block;
            margin-top: 20px;
            font-weight: bold;
            color: rgb(74, 74, 74);
        }

        input {
            width: 60%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }

        h3 {
            margin: 10px 0;
            color: #555;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #0575E6;
            color: white;
            margin: 10px;
            display: inline-block;
        }

        button:hover {
            background-color: #0060c1;
        }

        #error-message {
            color: red;
            display: none;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>Vamos a registrar tu voz</h1>
        <div id="alert-message" class="alert" style="display: none;"></div>
        <br>
        <br>
        <br>
        <!-- Eliminé el campo de ingreso de usuario -->
        <h3>Presione el botón "Iniciar grabación"</h3>
        <h3>Luego diga "Mi nombre es ...." y espera unos segundos.</h3>
        <i id="recordingIcon" class="fa-sharp fa-solid fa-microphone fa-beat fa-2xl" style="color: #0d59a0; display: none;"></i>
        <button id="startRecording" onclick="startRecording()">Iniciar grabación</button>
        <button id="stopRecording" style="display:none" onclick="stopRecording()">Detener grabación</button>
        <br>
        <br>
        <p id="error-message" style="color: rgb(255, 0, 21); display: none;">Usuario no existe. Por favor, regístrese antes de grabar el audio.</p>
    </div>

    <script>
        let recorder;
        let stopTimeout;

        function startRecording() {
            // Restablecer el mensaje de error al iniciar una nueva grabación
            document.getElementById('error-message').style.display = 'none';

            // Verificar la existencia de la sesión del usuario
            const username = "{{ session['user_id'] }}"; // Obtener el nombre de usuario de la sesión

            if (username) {
                // El usuario existe en la sesión, proceder con la lógica de grabación
                startRecordingLogic();
            } else {
                // Mostrar mensaje de error si no hay usuario en la sesión
                document.getElementById('error-message').style.display = 'block';
            }
        }

        function startRecordingLogic() {
            console.log('Iniciando grabación...');

            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    // Utiliza la configuración básica
                    recorder = new MediaRecorder(stream);

                    let audioChunks = [];

                    recorder.ondataavailable = function(e) {
                        if (e.data.size > 0) {
                            audioChunks.push(e.data);
                        }
                    };

                    recorder.onstop = function() {
                        let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        let formData = new FormData();
                        formData.append('audio', audioBlob, 'audio.wav');
                        formData.append('username', "{{ session['user_id'] }}"); // Enviar el nombre de usuario de la sesión

                        // Enviar el archivo al servidor
                        fetch('/uploadRegistrar', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            // Muestra el mensaje JSON devuelto por el servidor
                            console.log(data);
                            document.getElementById('alert-message').innerText = data.message;
                            document.getElementById('alert-message').style.display = 'block';
                        })
                        .catch(error => {
                            console.error('Error al enviar el archivo:', error);
                        });
                    };

                        recorder.start();
                    document.getElementById('startRecording').style.display = 'none';
                    document.getElementById('stopRecording').style.display = 'none';
                    document.getElementById('recordingIcon').style.display = 'inline-block'; // Muestra el ícono de grabación
                    
                    stopTimeout = setTimeout(function(){
                        stopRecording();
                        document.getElementById('recordingIcon').style.display = 'none';
                    }, 6000); // Detener la grabación después de 6 segundos
                })
                        .catch(function(err) {
                    console.error('Error al acceder al micrófono: ', err);
                });
        }

        function stopRecording() {
                recorder.stop();
                document.getElementById('startRecording').style.display = 'inline-block';
                document.getElementById('stopRecording').style.display = 'none';
                document.getElementById('startSession').style.display = 'inline-block';
            }
    </script>
</body>
</html>
{% endblock %}
