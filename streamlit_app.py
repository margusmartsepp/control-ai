import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import tiktoken

# Pricing per 1000 tokens (adjust according to the latest OpenAI pricing)
pricing = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "o1-preview": {"input": 0.015, "output": 0.06, "cached_input": 0.0075},
    "o1-preview-2024-09-12": {"input": 0.015, "output": 0.06, "cached_input": 0.0075},
    "o1-mini": {"input": 0.003, "output": 0.012, "cached_input": 0.0015},
    "o1-mini-2024-09-12": {"input": 0.003, "output": 0.012, "cached_input": 0.0015},
    "davinci-002": {"input": 0.012, "output": 0.012},
    "babbage-002": {"input": 0.0004, "output": 0.0004}
}

# Token limits for models
token_limits = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "davinci-002": 4096,
    "babbage-002": 4096
}

# Display title and description
st.title("💬 Chatbot with Token & Cost Tracker")
st.write("This chatbot tracks OpenAI token usage, costs, and allows you to manage the context by editing or deleting messages.")

# Ask for OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Allow model selection
model_choice = st.sidebar.selectbox("Choose OpenAI Model", list(token_limits.keys()))
max_tokens = token_limits[model_choice]

# Tokenizer function using OpenAI's tiktoken library
enc = tiktoken.encoding_for_model(model_choice)

def count_tokens(text):
    return len(enc.encode(text))

# Calculate cost function based on the model
def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * pricing[model_choice].get("input", 0)
    output_cost = (output_tokens / 1000) * pricing[model_choice].get("output", 0)
    cached_cost = (input_tokens / 1000) * pricing[model_choice].get("cached_input", 0)
    total_cost = input_cost + output_cost + cached_cost
    return total_cost

# Proceed if an API key is provided
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create a ChatOpenAI instance with Langchain, using the selected model
    chat_model = ChatOpenAI(api_key=openai_api_key, model_name=model_choice)

    # Create a session state variable for chat messages and token usage
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.token_usage = 0  # Track total token usage
        st.session_state.truncated_tokens = 0
        st.session_state.token_breakdown = []  # To store token count for each message

    # Function to recalculate token usage after any edit or delete
    def recalculate_token_usage():
        total_token_usage = 0
        st.session_state.truncated_tokens = 0
        st.session_state.token_breakdown = []  # Reset breakdown
        for message in st.session_state.messages:
            message_tokens = count_tokens(message["content"])
            total_token_usage += message_tokens
            st.session_state.token_breakdown.append(message_tokens)
            
            if total_token_usage > max_tokens:
                st.session_state.truncated_tokens += message_tokens
        st.session_state.token_usage = total_token_usage

    # Function to edit a message
    def edit_message(index):
        new_text = st.text_area(f"Edit message #{index + 1}", st.session_state.messages[index]["content"])
        if st.button(f"Save Edit for message #{index + 1}"):
            st.session_state.messages[index]["content"] = new_text
            recalculate_token_usage()

    # Function to delete a message
    def delete_message(index):
        st.session_state.messages.pop(index)
        recalculate_token_usage()

    # Display previous chat messages with Edit and Delete buttons
    total_token_usage = 0
    for i, message in enumerate(st.session_state.messages):
        message_tokens = count_tokens(message["content"])
        total_token_usage += message_tokens

        # Strikeout if token limit exceeded
        if total_token_usage > max_tokens:
            # If token limit exceeded, strike out the message
            with st.chat_message(message["role"]):
                st.markdown(f"~~{message['content']}~~")  # Strikethrough for truncated content
            st.session_state.truncated_tokens += message_tokens
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Show token count for each message
        st.write(f"Tokens used by Message #{i+1}: {message_tokens}")

        # Edit and Delete buttons for each message
        with st.expander(f"Message Actions for Message #{i+1}"):
            st.button(f"Edit Message #{i+1}", on_click=edit_message, args=(i,))
            st.button(f"Delete Message #{i+1}", on_click=delete_message, args=(i,))

    # Show total token usage so far
    st.write(f"**Total Tokens Used**: {st.session_state.token_usage}/{max_tokens} tokens")

    # Show token usage as a progress bar
    st.progress(total_token_usage / max_tokens, text=f"Token Usage: {total_token_usage}/{max_tokens} tokens")

    # Chat input from the user
    if prompt := st.chat_input("Type your message here..."):
        # Store and display the user's input
        prompt_tokens = count_tokens(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        total_token_usage += prompt_tokens
        st.session_state.token_breakdown.append(prompt_tokens)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare prompt template
        template = ChatPromptTemplate.from_template("{content}")
        chain = LLMChain(llm=chat_model, prompt=template)

        # Generate response using Langchain LLM
        response = chain.run({"content": prompt})
        response_tokens = count_tokens(response)
        total_token_usage += response_tokens
        st.session_state.token_breakdown.append(response_tokens)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Show token count for the input and response
        st.write(f"Tokens used by User Input: {prompt_tokens}")
        st.write(f"Tokens used by Assistant Response: {response_tokens}")

        # Calculate and display cost
        total_cost = calculate_cost(prompt_tokens, response_tokens)
        st.write(f"Cost of this interaction: ${total_cost:.4f}")

        # If token limit exceeded, show strikethrough
        if total_token_usage > max_tokens:
            st.warning("Context window exceeded. Some previous content is truncated.")
            st.session_state.truncated_tokens += prompt_tokens + response_tokens

    # Show token breakdown for each message
    st.write("### Token Breakdown for each message:")
    for idx, token_count in enumerate(st.session_state.token_breakdown):
        st.write(f"Message #{idx + 1}: {token_count} tokens")

    # Recalculate token usage whenever a message is edited or deleted
    recalculate_token_usage()
