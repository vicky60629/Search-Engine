import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
""")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Use API key from sidebar or fall back to environment variable
    groq_api_key = api_key if api_key else os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar or set it in your .env file.")
        st.stop()

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b", streaming=True)
    tools = [search, arxiv, wiki]

    # Modern prompt template for tool-calling agents
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research and search assistant. Use the tools provided when necessary to answer user questions accurately."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    search_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Format existing history for the agent executor (excluding the current prompt)
        chat_history_formatted = [
            (m["role"], m["content"]) for m in st.session_state.messages[:-1]
        ]
        
        response = search_agent.invoke(
            {
                "input": prompt,
                "chat_history": chat_history_formatted
            },
            callbacks=[st_cb]
        )
        
        output_text = response.get("output", "")
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.write(output_text)