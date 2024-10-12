import speech_recognition as sr
import librosa
import numpy as np
import joblib  # Import joblib directly (no longer from sklearn.externals)

# Load pre-trained model for accent classification
# Ensure this file exists and is in the correct directory
try:
    accent_classifier = joblib.load('accent_classifier_model.pkl')
except FileNotFoundError:
    print("Error: Accent classifier model file not found! Please check the file path.")

def explain_process():
    """Explains what the program does in simple terms."""
    print("This program will listen to your speech, transcribe it, and attempt to detect your accent.")
    print("Please speak clearly and wait for processing after you stop speaking.")

def configure_microphone(source, recognizer):
    """Configures the microphone by adjusting for background noise."""
    print("Adjusting microphone for background noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Microphone is ready. Please speak when you're ready!")

def extract_audio_features(audio_data, sample_rate=16000):
    """
    Extracts audio features (MFCC) from the captured audio.
    
    Args:
        audio_data (np.array): Audio data as a numpy array.
        sample_rate (int): The sample rate of the audio data.
        
    Returns:
        np.array: Mean of MFCC features for the given audio data.
    """
    try:
        print("Extracting audio features for accent detection...")
        # Extract MFCC (Mel-frequency cepstral coefficients) features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def detect_speech_and_accent():
    """
    Listens to the user's speech, converts it to text, and tries to detect the accent.
    """
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        # Explain what will happen
        explain_process()
        configure_microphone(source, recognizer)
        
        # Capture the speech
        print("Listening for your speech... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("Processing your speech...")

            # Recognize speech using Google's Speech Recognition API
            try:
                print("Transcribing your speech with Google...")
                speech_text = recognizer.recognize_google(audio)
                print(f"Transcription: {speech_text}")
                
                # Convert audio to numpy array for feature extraction
                audio_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                
                # Extract features for accent classification
                features = extract_audio_features(audio_np)
                
                # If feature extraction failed, skip the classification
                if features is None:
                    print("Could not extract audio features. Accent detection aborted.")
                    return
                
                # Predict accent using the pre-trained model
                print("Determining your accent...")
                predicted_accent = accent_classifier.predict([features])
                print(f"It seems like your accent is: {predicted_accent[0]}")
                
            except sr.UnknownValueError:
                print("Oops! I couldn't understand what you said.")
            except sr.RequestError as e:
                print(f"Error connecting to Google API: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during speech recognition: {e}")
        
        except sr.WaitTimeoutError:
            print("Hmm, it looks like you didn't say anything in time.")
        except sr.RequestError as e:
            print(f"There was an error accessing the microphone or audio: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while capturing your speech: {e}")

# Run the speech and accent detection function
detect_speech_and_accent()
