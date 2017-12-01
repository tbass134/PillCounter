import pyaudio
import wave
import time

formate = pyaudio.paInt24
channels = 1
framerate = 48000
fileName = 'test ' + '.wav'
chunk = 6144
# output of stream.get_read_available() at different positions

p = pyaudio.PyAudio()

stream = p.open(format=formate,
                channels=channels,
                rate=framerate,
                input=True,
                frames_per_buffer=chunk) # observation c

# get data
sampleList = []

for i in range(0, 79):
    data = stream.read(chunk)
    sampleList.append(data)

print('end -', time.strftime('%d.%m.%Y %H:%M:%S', time.gmtime(time.time())))

stream.stop_stream()
stream.close()
p.terminate()

# produce file
file = wave.open(fileName, 'w')
file.setnchannels(channels)
file.setframerate(framerate)
file.setsampwidth(p.get_sample_size(formate))
file.writeframes(b''.join(sampleList))
file.close()
