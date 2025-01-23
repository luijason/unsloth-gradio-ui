from ui import create_gradio_interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_port=8080)