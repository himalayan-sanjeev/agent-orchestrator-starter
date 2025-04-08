from crewai import Agent, Task, Crew
from langchain.chat_models import ChatLiteLLM
import os
from dotenv import load_dotenv

load_dotenv()

# Use the built-in LiteLLM wrapper (this is CrewAI-compatible!)
llm = ChatLiteLLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("LITELLM_API_KEY"),
    provider="gemini"
)

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Extract factual information and key points from provided documents",
    backstory="An expert at scanning text for relevant content",
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write structured summaries and paragraphs based on the extracted data",
    backstory="A talented technical writer who explains things clearly",
    verbose=True,
    llm=llm
)

reviewer = Agent(
    role="Reviewer",
    goal="Improve clarity, tone, and fix any errors in the generated content",
    backstory="An experienced editor who polishes writing for readability",
    verbose=True,
    llm=llm
)

def run_crew_for(topic):
    research_task = Task(
        description=f"Identify key facts and ideas about '{topic}'",
        expected_output="A bullet-point list of findings and insights",
        agent=researcher
    )

    writing_task = Task(
        description=f"Based on the research, write a short article about '{topic}'",
        expected_output="A 2-3 paragraph explanation in clear language",
        agent=writer,
        depends_on=[research_task]
    )

    review_task = Task(
        description="Review and improve the summary for grammar, clarity, and tone.",
        expected_output="A polished version of the summary.",
        agent=reviewer,
        depends_on=[writing_task]
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        verbose=True
    )

    return crew.kickoff()
