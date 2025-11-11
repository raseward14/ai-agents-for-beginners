import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core import CancellationToken

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

load_dotenv()
client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

agent = AssistantAgent(
    name="assistant",
    model_client=client,
    tools=[],
    system_message="You are a travel agent that plans great vacations",
)

async def assistant_run():
    # First test message
    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
    print("Capital of France result:", result)

    # Define the query
    user_query = "Plan me a great sunny vacation"
    print("User:", user_query)

    # Execute the agent response
    response = await agent.on_messages(
        [TextMessage(content=user_query, source="user")],
        cancellation_token=CancellationToken(),
    )

    print("Assistant:", response.chat_message.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(assistant_run())