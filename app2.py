import streamlit as st
from llm import chat_llm
import pandas as pd
import requests
import shutil
import os
import base64
# from deep_translator import GoogleTranslator
from translation import translator

current_dir = os.getcwd().replace("\\","/")
provider = "lmstudio"
model_path = "./mistral-7b-openorca.Q5_K_M.gguf"
dest_folder = current_dir + "/plots"

def save_image(image_input, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    def get_unique_path(path):
        base, extension = os.path.splitext(path)
        counter = 1
        unique_path = path
        while os.path.exists(unique_path):
            unique_path = f"{base}_{counter}{extension}"
            counter += 1
        return unique_path

    # Initialize the image path variable
    final_path = ""

    # Check if the input is a file path or a base64 string
    if os.path.isfile(image_input):
        # It's a file path
        final_path = os.path.join(destination_folder, os.path.basename(image_input))
        final_path = get_unique_path(final_path)
        shutil.copy(image_input, final_path)
    elif image_input.startswith("data:image") or image_input.startswith("iVBOR"):
        # It's a base64 string
        try:
            base64_string = image_input.split(",")[1]
        except IndexError:
            base64_string = image_input
        image_data = base64.b64decode(base64_string)
        final_path = os.path.join(destination_folder, "saved_image.png")
        final_path = get_unique_path(final_path)

        # Write the image data to a file
        with open(final_path, "wb") as file:
            file.write(image_data)
    else:
        print("Input is neither a valid file path nor a base64 string.")
        return None
    
    # Return the final path of the saved image
    return final_path

# Display message response
def display_response(message):
    if isinstance(message, pd.DataFrame):
        st.dataframe(message)
    elif isinstance(message, str) and any(substring in message for substring in [".png", "data:image/png;base64", "iVBOR"]):
        st.image(message, caption="Bot")
    else:
        st.write(f"**Bot:** {message}")

# Initialize the chat assistant and the model only once
if "chat_assistant" not in st.session_state:
    st.session_state.chat_assistant = chat_llm()
    st.session_state.translator = translator()
    st.session_state.df = st.session_state.chat_assistant.create_llm_chain(model_path, provider=provider)

# Set up the Streamlit app
st.title("Chatbot with LLM")
st.write("Enter your queries below and get responses from the model.")

# Initialize the session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # st.markdown(message["content"])
        display_response(message["content"])

# User input
if user_input := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Handle user input and model response
    try:
        translated_input = st.session_state.translator.to_eng(user_input)
        if isinstance(translated_input, list):
            translated_input = translated_input[0]
    except requests.exceptions.ConnectionError:
        st.write("**Bot:** Check Internet Connection for translation into English")
        st.session_state.messages.append({"role": "assistant", "content": "Check Internet Connection for translation into English"})
        exit()

    if provider == "lmstudio":
        response = st.session_state.df.chat(f"{translated_input}")
    else:
        response = st.session_state.df.llm.query(f"{translated_input}. return the final answer either in string or image according to the question.", yolo=True)
    
    if isinstance(response, pd.DataFrame):
        st.session_state.messages.append({"role": "assistant", "content": response})
    elif isinstance(response, str) and any(substring in response for substring in [".png", "data:image/png;base64", "iVBOR"]):
        response = save_image(response, dest_folder)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        display_response(response)


