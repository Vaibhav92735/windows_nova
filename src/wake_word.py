import pvporcupine
from pvrecorder import PvRecorder

porcupine = pvporcupine.create(access_key="6+33s0Ay0dEl/yeeWks0DoMrT7WOuPtt3639Sfm9d4wF92Bnu2iTuA==",
                               keyword_paths=["../models/Ok_nova_en_windows.ppn"])
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
recorder.start()

try:
    print("Listening for 'Ok Nova'...")
    while True:
        pcm = recorder.read()
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Wake word detected!")
            break
finally:
    recorder.stop()
    porcupine.delete()
    recorder.delete()
