import os

import matplotlib.pyplot as plt
from gtts import gTTS
from pydub import AudioSegment
from scipy.io import wavfile

from mathematics import *


def list_of_lists_to_1d_np_array(lst):
    return np.array(lst).flatten().astype(np.int16)


def visualize(segment, ttl="Cool Audio Waveform"):
    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(segment)) / len(segment) / 50.0, segment, color='green')
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time (seconds)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title(ttl, color='white')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    txt = input("Enter small text: ")
    wow = gTTS(text=txt, lang='ja', slow=False)
    file_name = "Sample"
    wow.save(file_name + ".mp3")
    sound = AudioSegment.from_mp3(f"{file_name}.mp3")
    sound.export(f"{file_name}.wav", format="wav")
    os.remove(f"./{file_name}.mp3")
    bit_rate, data = wavfile.read(f"./{file_name}.wav")

    data = [j for i, j in enumerate(data) if i % 2]
    segments = []
    n = len(list(data)) // 882
    for i in range(0, n - 1):
        segments.append(data[882 * i: 882 * (i + 1)])

    # Next 4 lines are for report purposes: visualize(segment) to plot the curve of segment [10] for all 3 interpolation methods
    visualize(segments[10])
    visualize(bezier_for_segment(segments[10]), "Bézier Curve")
    visualize(catmull_rom_for_segment(segments[10]), "Catmull-Rom Spline")
    visualize(lagrange_interpolation(segments[10]), "Lagrange Interpolation")

    beziered_segments = []
    catmul_rommed_segments = []
    lagrange_segments = []
    for segment in segments:
        beziered_segments.append(bezier_for_segment(segment))
        catmul_rommed_segments.append(catmull_rom_for_segment(segment))
        lagrange_segments.append(lagrange_interpolation(segment))


    # Turn beziered_segments, catmull_rommed_segments and lagrange_segments into wav files
    beziered_segments = list_of_lists_to_1d_np_array(beziered_segments)
    catmul_rommed_segments = list_of_lists_to_1d_np_array(catmul_rommed_segments)
    lagrange_segments = list_of_lists_to_1d_np_array(lagrange_segments)

    wavfile.write(filename=f"{file_name}_bezier.wav", data=beziered_segments, rate=bit_rate)
    wavfile.write(filename=f"{file_name}_catmull_rom.wav", data=catmul_rommed_segments, rate=bit_rate)
    wavfile.write(filename=f"{file_name}_lagrange.wav", data=lagrange_segments, rate=bit_rate)
