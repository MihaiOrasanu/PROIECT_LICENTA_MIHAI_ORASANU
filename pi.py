import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import random
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from scipy.signal import butter, filtfilt

# Define the class labels
class_labels = ['engine_idling_voce', 'engine_idling', 'siren_voce', 'siren', 'jackhammer_voce', 'jackhammer']

def display_message(message):
    print(message)

def record_audio(filename, durata=10, f=44100):
    display_message("Recording started...")
    audio = sd.rec(int(durata * f), samplerate=f, channels=2)
    sd.wait()  # Wait until recording is finished
    sf.write(filename, audio, f)
    display_message(f"Recording saved to {filename}")

def load_wav(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def resample_wav(y, sr, target_sr=44100):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def overlap_wav(file1, file2, output_file1='overlapped.wav', output_file2='noise_extended.wav'):
    y1, sr1 = load_wav(file1)
    y2, sr2 = load_wav(file2)

    if sr1 != 44100:
        y1, sr1 = resample_wav(y1, sr1, 44100)
    if sr2 != 44100:
        y2, sr2 = resample_wav(y2, sr2, 44100)

    if len(y1) > len(y2):
        y2 = np.tile(y2, int(np.ceil(len(y1) / len(y2))))[:len(y1)]
    else:
        y1 = np.tile(y1, int(np.ceil(len(y2) / len(y1))))[:len(y2)]

    overlapped = y1 + y2
    sf.write(output_file1, overlapped, sr1)
    sf.write(output_file2, y2, sr2)
    display_message(f"Overlapped WAV saved to {output_file1} and extended noise saved to {output_file2}")

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

def classify_spectrogram(image_file, base_model, interpreter, input_details, output_details):
    # Load and preprocess the image
    x = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features using MobileNetV2
    y = base_model.predict(x)

    # Reshape the features to match the expected input shape of the TFLite model
    input_shape = input_details[0]['shape']
    y = y.reshape(input_shape)

    # Set the tensor to the TFLite model
    interpreter.set_tensor(input_details[0]['index'], y)
    interpreter.invoke()

    # Get the output from the TFLite model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print classification results
    for i, label in enumerate(class_labels):
        print(f'{label}: {output_data[0][i]}')

    # Find the label with the highest score
    max_index = np.argmax(output_data[0])
    return class_labels[max_index]

def butter_filter(data, cutoff, fs, filter_type, order=4):
    f = 0.5 * fs
    normal_cutoff = cutoff / f
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    y = filtfilt(b, a, data)
    return y

def passive_filter(class_label, x1, x2, fs):
    if class_label in ['engine_idling_voce', 'jackhammer_voce']:
        cutoff_freq = 80  # Highpass filter at 80 Hz
        x1_filtered = butter_filter(x1, cutoff_freq, fs, 'high')
        x2_filtered = butter_filter(x2, cutoff_freq, fs, 'high')
    elif class_label == 'siren_voce':
        f0 = 800  # Bandstop filter centered at 800 Hz
        bw = 200  # Bandwidth of 200 Hz
        Wn = [(f0 - bw/2) / (fs / 2), (f0 + bw/2) / (fs / 2)]
        b, a = butter(4, Wn, btype='bandstop')
        x1_filtered = filtfilt(b, a, x1)
        x2_filtered = filtfilt(b, a, x2)
    else:
        attenuation = 10 ** (-6 / 20)  # 6 dB attenuation
        x1_filtered = x1 * attenuation
        x2_filtered = x2 * attenuation
    return x1_filtered, x2_filtered

def lms_filter(desired, noise, mu=0.01, filter_order=32):
    n = len(desired)
    w = np.zeros(filter_order)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(filter_order, n):
        x = noise[i:i - filter_order:-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w = w + 2 * mu * e[i] * x
    return e

def snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

def adaptive_filter(input_file, noise_file, output_file='filtered.wav'):
    desired, sr_desired = load_wav(input_file)
    noise, sr_noise = load_wav(noise_file)

    if sr_desired != sr_noise:
        raise ValueError("Sample rates of the two audio files do not match")

    # Calculate and display SNR before filtering
    snr_before = snr(desired, noise)
    print(f"SNR before filtering: {snr_before:.2f} dB")

    # Apply LMS filter
    filtered_signal = lms_filter(desired, noise)

    # Calculate and display SNR after filtering
    snr_after = snr(desired, filtered_signal)
    print(f"SNR after filtering: {snr_after:.2f} dB")

    # Save the filtered signal
    sf.write(output_file, filtered_signal, sr_desired)
    display_message(f"Filtered WAV saved to {output_file}")

def choose_random_wav(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    if not files:
        raise ValueError(f"No WAV files found in folder: {folder}")
    return os.path.join(folder, random.choice(files))

def main():
    # Load the pretrained models
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Load the custom model
    model_path = 'model.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    project_folder = os.path.dirname(os.path.abspath(__file__))
    record_audio(os.path.join(project_folder, 'recorded.wav'), durata=10)

    print("Folders available:")
    for folder in next(os.walk(project_folder))[1]:
        print(folder)

    chosen_folder = input("Enter the name of the folder to choose a random WAV file from: ")
    chosen_folder_path = os.path.join(project_folder, chosen_folder)
    chosen_wav = choose_random_wav(chosen_folder_path)
    print(f"Randomly chosen WAV file: {chosen_wav}")

    overlap_wav(os.path.join(project_folder, 'recorded.wav'), chosen_wav)

    # Create a spectrogram
    create_spectrogram(os.path.join(project_folder, 'overlapped.wav'), os.path.join(project_folder, 'overlapped.png'))

    # Classify the spectrogram
    class_label = classify_spectrogram(os.path.join(project_folder, 'overlapped.png'), base_model, interpreter, input_details, output_details)
    print(f'Predicted class: {class_label}')

    # Load overlapped audio
    overlapped, sr_overlapped = load_wav(os.path.join(project_folder, 'overlapped.wav'))
    noise_extended, sr_noise = load_wav(os.path.join(project_folder, 'noise_extended.wav'))

    # Apply passive filtering
    overlapped_filtered, noise_filtered = passive_filter(class_label, overlapped, noise_extended, sr_overlapped)
    sf.write(os.path.join(project_folder, 'overlapped_filtered.wav'), overlapped_filtered, sr_overlapped)
    sf.write(os.path.join(project_folder, 'noise_filtered.wav'), noise_filtered, sr_noise)
    display_message("Passive filtering applied and files saved.")

    # Apply adaptive filtering
    adaptive_filter(os.path.join(project_folder, 'overlapped_filtered.wav'), os.path.join(project_folder, 'noise_filtered.wav'))

if __name__ == "__main__":
    main()
