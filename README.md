# Chatbot with LLM

This project implements a chatbot using Large Language Models (LLMs) integrated with Streamlit for an interactive user interface. The bot is capable of handling text inputs, performing language translation, and displaying both text and image responses. This bot takes user query and then generates a graph or an appropriate answer for the user.

## Features

- **Interactive UI**: Utilizes Streamlit for an engaging and interactive chat experience.
- **Language Translation**: Translates user input to English before processing (requires internet connection).
- **LLM Integration**: Connects to a language model for generating responses.
- **Image Handling**: Supports displaying image responses and saving base64-encoded images to a specified directory.
- **Session Management**: Maintains conversation history within the session state.

## Requirements

- Python 3.10+

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/ShkAmmarHussain/Postgres-Visualise-Chat-bot.git
    cd Postgres-Visualise-Chat-bot
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your environment variables:

    ```plaintext
    HF_LOGIN_TOKEN=""
    DB_NAME=""
    DB_USER=""
    DB_PASSWORD=""
    DB_HOST=host.docker.internal
    DB_TABLE=""
    LLM_PROVIDER=""
    LLM_MODEL=""
    ```

5. Ensure your language model is correctly set up and accessible at the specified path.

## Configuration

Modify the following variables in the script as per your setup (if not using environment variables):

- `provider`: Specify your LLM provider.
- `model_path`: Path to your language model.
- `dest_folder`: Directory to save images.

```python
provider = os.getenv('LLM_PROVIDER', 'lmstudio')
model_path = os.getenv('LLM_MODEL', './mistral-7b-openorca.Q5_K_M.gguf')
dest_folder = current_dir + "/plots"
```

## Running the App

To run the Streamlit app, execute the following command:

```bash
streamlit run app.py
```

## Usage

1. Open your browser and navigate to the local URL provided by Streamlit.
2. Enter your queries in the input box.
3. View responses from the chatbot, which may include text or images.
4. The conversation history will be displayed in the chat interface.

## Code Explanation

### Main Functions

1. **save_image(image_input, destination_folder)**: Saves images from file paths or base64 strings to the specified directory.

2. **display_response(message)**: Displays the response message, which could be text, a DataFrame, or an image.

### Initialization

- Initializes the chat assistant and model only once using `st.session_state`.
- Sets up the conversation history and user interface.

### User Interaction

- Handles user input and translation to English.
- Queries the language model for a response.
- Displays the response and updates the conversation history.

## Note

- Ensure an internet connection for translation functionality.
- Modify the script according to your LLM provider's API and requirements.

## Troubleshooting

- **Connection Issues**: Ensure you have a stable internet connection for translation.
- **Model Path**: Verify the path to your language model is correct and accessible.
- **Dependencies**: Ensure all required packages are installed.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

Special thanks to the contributors and the open-source community for their valuable resources and support.
