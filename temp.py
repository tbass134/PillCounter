import pyaudio
p = pyaudio.PyAudio()
devinfo = p.get_device_info_by_index(1)  # Or whatever device you care about.
print(devinfo)
if p.is_format_supported(44100.0,  # Sample rate
                         input_device=devinfo['index'],
                         input_channels=1,
                         input_format=pyaudio.paInt16):
  print 'Yay!'
