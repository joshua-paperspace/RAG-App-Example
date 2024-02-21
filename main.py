from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

app = FastAPI()

async def generate():
    inputs = tokenizer(["An increasing sequence: one,"], return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=50)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

@app.get("/")
async def main():
    return StreamingResponse(generate())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)