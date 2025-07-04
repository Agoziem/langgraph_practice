# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import Annotated, TypedDict, cast
from langgraph.checkpoint.memory import MemorySaver
import uuid
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[list[str], add_messages]
    human_feedback: Annotated[list[str], add_messages]


def model_node(state: State):
    print("Generating LinkedIn post...")
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else [
        "no feedback yet"]

    prompt = f"""
    linkedin_topic: {linkedin_topic}
    human_feedback: {feedback[-1] if feedback else "no feedback yet"}

    Generate a Structured and well written LinkedIn post based on the topic and feedback.
    The post should be engaging, professional, and suitable for a LinkedIn audience.

    consider previous feedback to improve the post.
    """

    response = llm.invoke([SystemMessage(content="You are a expert LinkedIn content generator."),
                           HumanMessage(content=prompt)])

    generated_post = response.content
    print(f"Generated LinkedIn post: {generated_post}")

    return {
        "generated_post": [AIMessage(content=generated_post)],
        "human_feedback": feedback
    }


def human_node(state: State):
    """Human feedback node for LinkedIn post generation. loops back to the model node."""
    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post": generated_post,
            "message": "Please provide feedback on the generated LinkedIn post. or type 'done' to finish.",
        }
    )

    print(f"User feedback: {user_feedback}")

    if user_feedback.lower() == "done":
        return Command(
            update={
                "human_feedback": state["human_feedback"] + ["finalized"],

            },
            goto="end_node"
        )

    return Command(
        update={
            "human_feedback": state["human_feedback"] + [HumanMessage(content=user_feedback)],
        },
        goto="model_node"
    )


def end_node(state: State):
    """End node to finalize the LinkedIn post generation."""
    print("Finalizing LinkedIn post generation...")
    final_post = state["generated_post"][-1] if state["generated_post"] else "No post generated."
    print(f"Final LinkedIn post: {final_post}")
    print("Final Human feedback:", state["human_feedback"])

    return {
        "generated_post": state["generated_post"],
        "human_feedback": state["human_feedback"],
    }


# Define the state graph
state_graph = StateGraph(State)
state_graph.add_node("model_node", model_node)
state_graph.add_node("human_node", human_node)
state_graph.add_node("end_node", end_node)

# Define the edges between nodes
state_graph.add_edge(START, "model_node")
state_graph.add_edge("model_node", "human_node")

state_graph.set_finish_point("end_node")

checkpointer = MemorySaver()
graph = state_graph.compile(checkpointer=checkpointer)

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# Enable the interrupt feature
thread_config = RunnableConfig(
    {"configurable": {
        "thread_id": str(uuid.uuid4()),
    }}
)

linkedin_topic = input("Enter the LinkedIn topic: ")
initial_state = {
    "linkedin_topic": linkedin_topic,
    "generated_post": [],
    "human_feedback": [],
}

for chunk in graph.stream(
    initial_state,
    config=thread_config,
):
    for node_id, value in chunk.items():
        if node_id == "__interrupt__":
            while True:
                user_input = input(
                    "provide feedback or type 'done' to finish: ")
                # resume the graph execution with the user input
                graph.invoke(Command(resume=user_input),
                             config=thread_config)

                # Exit loop if user says done
                if user_input.lower() == "done":
                    break