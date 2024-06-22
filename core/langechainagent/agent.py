from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage

from langgraph.checkpoint.sqlite import SqliteSaver


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools,system="") -> None:
        self.system = system
        
        state_graph = StateGraph(AgentState)
        state_graph.add_node("llm", self.call_openai)
        state_graph.add_node("action", self.take_action)
        state_graph.add_conditional_edges(
            "llm", self.exists_action,
            {True: "action", False: END}
        )

        state_graph.add_edge("action", "llm")
        state_graph.set_entry_point("llm")

        memory = SqliteSaver.from_conn_string(":memory:")

        self.graph = state_graph.compile(checkpointer=memory, debug=True)
        self.tools = {t.name: t for t in tools}
        # bind tools to llm model
        self.model = model.bind_tools(tools)

        

    # llm node
    def call_openai(self, state:AgentState):
        print("Calling LLM Model")
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        messages = self.model.invoke(messages)
        return {'messages':[messages]}
    

    # action node
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for tool in tool_calls:
            print(f"calling tool id {tool['id']} , Name {tool['name']} with arguments {tool['args']}")
            if not tool['name'] in self.tools: # check if tool is bad from LLM
                print("\n...bad tool name..") 
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[tool['name']].invoke(tool['args'])
            results.append(ToolMessage(tool_call_id=tool['id'], name=tool['name'], content=str(result)))

        print('Action Node completed, Back to the model!')
        return {'messages': results}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0