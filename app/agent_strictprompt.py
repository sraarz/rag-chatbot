from app.llm import llm
from app.graph import graph

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.tools.vector import get_movie_plot
from app.tools.cypher import cypher_qa
from app.utils import get_session_id

# =========================================================
# 0) General chat tool (LLM-only) — deterministic + clean
# =========================================================
# If your llm object supports bind(), this makes evaluation more stable.
# If not supported, remove `.bind(...)`.
try:
    llm_eval = llm.bind(temperature=0)
except Exception:
    llm_eval = llm

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)
movie_chat = chat_prompt | llm_eval | StrOutputParser()


# =========================================================
# 1) Tools — add strict JSON requirement in tool descriptions 
# =========================================================
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    ), 
    Tool.from_function(
        name="Movie Plot Search (Vector)",    ## vector search for similarity finding
        description="For when you need to find information about movies based on a plot",
        func=get_movie_plot, 
    ),
    Tool.from_function(
        name="Movie information (Graph)",
        description="Provide information about movies questions using Cypher",
        func = cypher_qa
    )
]

# tools = [
#     Tool.from_function(
#         name="General Chat",
#         description=(
#             "Use for general movie discussion, concepts, and advice that does NOT require database lookup "
#             "or plot similarity search. Do not call this tool if the user asks for specific factual DB fields "
#             "(director/year/genres/cast for a known title) or plot-based identification."
#         ),
#         func=movie_chat.invoke,
#     ),
#     Tool.from_function(
#         name="Movie Plot Search (Vector)",  # vector search for plot similarity finding
#         description=(
#             "Use when the user provides a plot or plot-like description and you must identify a movie. "
#             "Action Input MUST be valid JSON, exactly like: "
#             '{"query": "<plot text>"}'
#         ),
#         func=get_movie_plot,
#     ),
#     Tool.from_function(
#         name="Movie information (Graph)",
#         description=(
#             "Use for factual questions about a known movie title using Cypher/graph DB "
#             "(director, year, genres, cast, etc.). "
#             "Action Input MUST be valid JSON, exactly like: "
#             '{"title": "<movie title>"} '
#             "Optionally: "
#             '{"title": "<movie title>", "year": 1997}'
#         ),
#         func=cypher_qa,
#     ),
# ]


# =========================================================
# 2) Memory
# =========================================================
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)



# =========================================================
# 3) Stricter ReAct prompt
#    - prevents 'Input:' instead of 'Action Input:'
#    - forbids mixing Action + Final Answer in one message
#    - enforces valid JSON for Action Input
# =========================================================
agent_prompt = PromptTemplate.from_template(r"""
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

IMPORTANT: Do not answer any questions using your pre-trained knowledge.
You MUST rely only on (1) tool outputs (Observations) and/or (2) the provided conversation history.

========================
STRICT OUTPUT FORMAT
========================
You MUST output EXACTLY ONE of the following two formats.

(A) TOOL CALL (and nothing else):
Thought: Do I need to use a tool? Yes
Action: <one of [{tool_names}]>
Action Input: <VALID JSON ONLY>

Rules for (A):
- Do NOT include Observation in your message. Observation will be provided by the system.
- Do NOT add any extra lines, bullets, explanations, or markdown.
- Action Input MUST be valid JSON (double quotes, proper braces).

(B) FINAL RESPONSE (and nothing else):
Thought: Do I need to use a tool? No
Final Answer: <your response here>

Rules for (B):
- Final Answer MUST be in the same message as the Thought line.
- Do NOT include Action/Action Input/Observation in the final response message.
- You may format the Final Answer with short sentences or bullet points, but it MUST start with "Final Answer:".

========================
TOOL ROUTING POLICY
========================
- If the user provides a plot/story description and wants to identify a movie: use "Movie Plot Search (Vector)".
- If the user asks factual fields about a known movie title (actors/cast, director, year, genres, etc.): use "Movie information (Graph)".
- Otherwise: answer directly using format (B).

TOOLS:
------
{tools}

Previous conversation history:
{chat_history}

New input:
{input}

{agent_scratchpad}
""".strip())

# agent_prompt = PromptTemplate.from_template(r"""
# You are a movie expert providing information about movies.

# Hard constraints (must follow):
# 1) Only answer movie-related questions (movies, plots, actors, directors, genres, release year).
# 2) NEVER use your pre-trained knowledge. You MUST rely only on tool outputs (Observations) and/or the conversation history provided.
# 3) Tool calling format is STRICT. If you decide to use a tool, you MUST output EXACTLY these lines and nothing else:
#    Thought: Do I need to use a tool? Yes
#    Action: <one of [{tool_names}]>
#    Action Input: <VALID JSON ONLY>
# 4) NEVER write "Input:" — always write "Action Input:" exactly.
# 5) NEVER include "Final Answer:" in the SAME message where you call a tool.
#    - Tool call message contains ONLY Thought/Action/Action Input (no extra text).
#    - After you receive an Observation, you may produce the Final Answer in a separate message.
# 6) Action Input MUST be VALID JSON (double quotes, proper braces). Examples:
#    - Movie Plot Search (Vector): {{"query": "plot text here"}}
#    - Movie information (Graph): {{"title": "Movie Title"}}
#      Optionally: {{"title": "Movie Title", "year": 1997}}

# Decision policy (tool routing):
# - If user asks about plot/story description / "find this movie by plot": use Movie Plot Search (Vector).
# - If user asks factual fields about a specific movie title (director/year/genres/cast): use Movie information (Graph).
# - Otherwise (general cinema talk / definitions / advice): do NOT call tools and answer directly.

# TOOLS:
# ------
# {tools}

# Previous conversation history:
# {chat_history}

# New input:
# {input}

# {agent_scratchpad}
# """.strip())



# =========================================================
# 4) Create agent + executor
#    Extra: return_intermediate_steps helps debugging eval.
# =========================================================
agent = create_react_agent(llm_eval, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,          
    return_intermediate_steps=True,      # helpful for evaluation/debugging
)


# =========================================================
# 5) Wrap with message history (required because prompt expects {chat_history})
# =========================================================
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']