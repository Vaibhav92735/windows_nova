import whisper

# Load Whisper (model choice: tiny/base/small/medium)
model = whisper.load_model("small")
# Suppose you saved audio to 'input.wav'; this transcribes it.
result = model.transcribe("input.wav")
text = result["text"]
print("User said:", text)
