from ui import create_gradio_interface

# if __name__ == "__main__":
#     demo = create_gradio_interface()
#     demo.launch(server_port=8080)

from fastapi import FastAPI
import gradio as gr

app = FastAPI()
demo = create_gradio_interface()
app = gr.mount_gradio_app(app, demo, path="/")