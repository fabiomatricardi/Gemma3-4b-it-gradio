import gradio as gr
from openai import OpenAI
import subprocess
import os
import signal
from time import sleep
import threading

# when using llamacpp-server, you need to check if the stream chunk is present
# usually the first and the last chunk are empty and will throw an error
# https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

# Global variable to store the process
gemmaServer = None

def start_server():
    global gemmaServer
    
    # Check if process is already running
    if gemmaServer is not None and gemmaServer.poll() is None:
        return "Server is already running!"
    
    # Start the server process
    gemmaServer = subprocess.Popen([
        'llama-server.exe',
        '-m',
        'google_gemma-3-4b-it-Q6_K.gguf',
        '-c',         
        '8196',
        '-ngl',
        '0',   
        '--port',
        '8080',       
    ], creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    return f"Server started with PID: {gemmaServer.pid}"

def delayed_exit():
    # Give the interface time to send the response before exiting
    import time
    time.sleep(2)
    os._exit(0)  # Force exit the Python process

def stop_server():
    global gemmaServer
    
    if gemmaServer is None:
        return "No server is running!"
    
    if gemmaServer.poll() is None:  # Check if process is still running
        try:
            gemmaServer.terminate()
            # Wait for process to terminate (optional)
            gemmaServer.wait(timeout=5)
            return "Server stopped successfully!"
        except subprocess.TimeoutExpired:
            gemmaServer.kill()
            return "Server killed forcefully!"
        except Exception as e:
            return f"Error stopping server: {str(e)}"
    else:
        return "Server is not running!"



def init_shutdown():
    # Schedule a forced exit after showing the message
    threading.Thread(target=delayed_exit, daemon=True).start()
    return " Closing Gradio interface..."


example = """
#### Example for Image Generation help
"""
mycode ="""
```
I want to create an image with Flux but I need assistance for a good prompt. 
The image should be about '''[userinput]'''. Comic art style.
```
"""
note = """#### 🔹 Gemma 3 4B Instruct
> [Gemma 3](https://ai.google.dev/gemma/docs/core), a collection of lightweight, state-of-the-art open models built from the same research and technology that powers our Gemini 2.0 models. 
<br>

[Gemma 3](https://developers.googleblog.com/en/introducing-gemma3/) comes in a range of sizes (1B, 4B, 12B and 27B)
These are the Google most advanced, portable and responsibly developed open models yet. 
<br><br>

Starting settings: `Temperature=0.45` `Max_Length=1500`
"""

# STARTING THE INTERFACE
with gr.Blocks(theme=gr.themes.Ocean()) as demo: #gr.themes.Ocean() Citrus() #https://www.gradio.app/guides/theming-guide
    gr.Markdown("# Chat with Gemma 3 4b Instruct 🔷 running Locally with [llama.cpp](https://github.com/ggml-org/llama.cpp)")
    with gr.Row():
        with gr.Column(scale=1):
            start_btn = gr.Button("Start Model Server",variant='primary')
            stop_btn = gr.Button("Stop Model Server") 
            srv_stat = gr.Textbox(label="Status")           
            maxlen = gr.Slider(minimum=250, maximum=4096, value=1500, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.45, step=0.1, label="Temperature")          
            gr.Markdown(note)
            closeall = gr.Button("Close the app",variant='stop')
            with gr.Accordion("See suggestions",open=False):
                gr.Markdown(example)
                gr.Code(mycode,language='markdown',wrap_lines=True)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages",show_copy_button = True,
                    avatar_images=['https://icons.iconarchive.com/icons/artua/dragon-soft/512/User-icon.png',''
                    'https://clipartcraft.com/images/transparent-background-google-logo-brand-2.png'],
                    height=550, layout='bubble')
            msg = gr.Textbox(lines=3,placeholder='Shift+Enter to send your message')
            # Button the clear the conversation history
            clear = gr.ClearButton([msg, chatbot],variant='primary')
    # Handle the User Messages
    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]    
    # HANDLE the inference with the API server
    def respond(chat_history,t,m):
        STOPS = ['<eos>']
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed", organization='Gemma3')
        stream = client.chat.completions.create(     
            messages=chat_history,
            model='Gemma 3 4B Instruct',
            max_tokens=m,
            stream=True,
            temperature=t,
            stop=STOPS)
        chat_history.append({"role": "assistant", "content": ""})
        for chunk in stream:
            # this is used with llama-server
            if chunk.choices[0].delta.content:
                chat_history[-1]['content'] += chunk.choices[0].delta.content
            yield chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(respond, [chatbot,temperature,maxlen], [chatbot])
    start_btn.click(start_server, inputs=[], outputs=srv_stat)
    stop_btn.click(stop_server, inputs=[], outputs=srv_stat) 
    closeall.click(stop_server, inputs=[], outputs=srv_stat).then(init_shutdown,inputs=[], outputs=srv_stat)   
# LAUNCH THE GRADIO APP with Opening automatically the default browser
demo.launch(inbrowser=True)
