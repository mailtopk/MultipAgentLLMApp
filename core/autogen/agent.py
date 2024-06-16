from autogen import ConversableAgent
from utils import helper

class Agent(ConversableAgent):
    """Represents an agent capable of conversing with users or an other agent."""

    def __init__(self, agent_name:str, system_message:str):
        """
        Initialize the Agent.

        Args:
            agent_name (str): The name of the agent.
            system_message (str): Prompt string.
        """                
        super().__init__(name=agent_name, 
                         system_message=system_message,
                         human_input_mode="NEVER",
                         llm_config=self.get_llm_config())


    def get_llm_config(self) -> dict:
        """
        Get the configuration for the large language model.

        Args:
            None
        Returns:
            dict: A dictionary containing the model and Open API key.
        """                
        key, model = helper.get_open_ai_model_and_key()
        return {"model" : model, "api_key" : key}   
    
