from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool


def create_agent(role: str, goal: str, backstory: str, delegate: bool) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        allow_delegation=delegate,
        verbose=True
    )

def build_crew():
    # create agents
    support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful "
            "support representative in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and "
            "are now working on providing "
            "support to {customer}, a super important customer "
            "for your company."
            "You need to make sure that you provide the best support!"
            "Make sure to provide full complete answers, "
            "and make no assumptions."
        ),
        allow_delegation=False,
        verbose=True
    )


    support_quality_assurance_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the "
            "best support quality assurance in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and "
            "are now working with your team "
            "on a request from {customer} ensuring that "
            "the support representative is "
            "providing the best support possible.\n"
            "You need to make sure that the support representative "
            "is providing full"
            "complete answers, and make no assumptions."
        ),
        verbose=True
    )

    # build tools
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    docs_scrape_tool = ScrapeWebsiteTool(website="https://docs.crewai.com/quickstart")

    # create tasks
    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
            "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
            "Make sure to use everything you know "
            "to provide the best support possible."
            "You must strive to provide a complete "
            "and accurate response to the customer's inquiry."
        ),
        expected_output=(
            "A detailed, informative response to the "
            "customer's inquiry that addresses "
            "all aspects of their question.\n"
            "The response should include references "
            "to everything you used to find the answer, "
            "including external data or solutions. "
            "Ensure the answer is complete, "
            "leaving no questions unanswered, and maintain a helpful and friendly "
            "tone throughout."
        ),
        tools=[docs_scrape_tool],
        agent=support_agent
    )

    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Senior Support Representative. "
            "Ensure that the answer is comprehensive, accurate, and adheres to "
            "high-quality standards expected for customer support.\n"
            "Verify that all parts of the customer's inquiry "
            "have been addressed "
            "thoroughly, with a helpful and friendly tone.\n"
            "Check for references and sources used to "
            "find the information, "
            "ensuring the response is well-supported and "
            "leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response "
            "ready to be sent to the customer.\n"
            "This response should fully address the "
            "customer's inquiry, incorporating all "
            "relevant feedback and improvements.\n"
            "Don't be too formal, we are a chill and cool company "
            "but maintain a professional and friendly tone throughout."
        ),
        agent=support_quality_assurance_agent,
    )



    # build crews
    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=True,
        memory=True
    )

    return crew


if __name__ == "__main__":
    crew = build_crew()
    result = crew.kickoff(inputs = {
            "customer": "DeepLearningAI",
            "person": "Andrew Ng",
            "inquiry": "I need help with setting up a Crew "
                    "and kicking it off, specifically "
                    "how can I add memory to my crew? "
                    "Can you provide guidance?"
        }
    )
    print(result)

    