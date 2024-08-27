import whisper

model = whisper.load_model("base")
result = model.transcribe(r'/podcasts-mp3/maquinas')

print(result["text"])