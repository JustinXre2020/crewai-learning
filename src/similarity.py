from crewai import Agent, Task, Crew
from crewai_tools import RagTool
from textblob import TextBlob
from pydantic import BaseModel, Field, validator
from typing import Dict


class SentimentAnalysis(BaseModel):
    """
    Pydantic model representing the sentiment analysis results.
    """
    sentiment: str = Field(
        ...,
        description="Overall sentiment classification (Positive, Negative, or Neutral)"
    )
    polarity_score: float = Field(
        ...,
        description="Sentiment polarity score from -1.0 (very negative) to 1.0 (very positive)",
        ge=-1.0,
        le=1.0
    )
    subjectivity_score: float = Field(
        ...,
        description="Sentiment subjectivity score from 0.0 (objective) to 1.0 (subjective)",
        ge=0.0,
        le=1.0
    )
    analysis: Dict[str, str] = Field(
        ...,
        description="Descriptions of the analysis metrics"
    )
    text_sample: str = Field(
        ...,
        description="Sample of the analyzed text"
    )

    @validator('sentiment')
    def validate_sentiment(cls, v):
        allowed_values = ["Positive", "Negative", "Neutral"]
        if v not in allowed_values:
            raise ValueError(f"Sentiment must be one of {allowed_values}")
        return v

class SentimentAnalysisTool(RagTool):
    name: str = "Sentiment Analysis Tool"
    description: str = "A tool that analyzes the sentiment of a given text."

    def _run(self, text: str) -> str:
        """
        Analyzes the sentiment of the given text using TextBlob.
        
        Returns:
            A string containing formatted sentiment analysis results including:
            - Polarity: -1.0 (very negative) to 1.0 (very positive)
            - Subjectivity: 0.0 (objective) to 1.0 (subjective)
            - Overall sentiment classification
        """
        analysis = TextBlob(text)
        
        # Calculate polarity and subjectivity
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.2:
            sentiment = "Positive"
        elif polarity < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        # Format the results
        sentiment_result = SentimentAnalysis(
            sentiment=sentiment,
            polarity_score=round(polarity, 2),
            subjectivity_score=round(subjectivity, 2),
            analysis={
                "polarity": "Ranges from -1.0 (very negative) to 1.0 (very positive)",
                "subjectivity": "Ranges from 0.0 (objective) to 1.0 (subjective)"
            },
            text_sample=text[:100] + "..." if len(text) > 100 else text
        )
        
        # Return the object
        return sentiment_result.model_dump_json()


def build_crew() -> Crew:
    senti_tool = SentimentAnalysisTool()

    # create agents
    analyze_agent = Agent(
        role="Language Analysis Expert",
        goal="Analyze and interpret text data to extract insights",
        backstory=(
            "As a Language Analysis Expert, "
            "you delve into the nuances of language, "
            "extracting insights from user input: {text} "
            "Your expertise in sentiment analysis and text interpretation "
            "enables you to provide valuable feedback "
            "on communication strategies and content effectiveness."
        ),
        allow_delegation=False,
        verbose=True
    )

    analyze_task = Task(
        description="Analyze the sentiment of the provided text.",
        expected_output="A JSON object containing sentiment analysis results.",
        tools=[senti_tool],
        output_json=SentimentAnalysis,
        agent=analyze_agent,
    )

    # build crew
    crew = Crew(
        agents=[analyze_agent],
        tasks=[analyze_task],
        verbose=True,
        memory=True
    )
    return crew


if __name__ == "__main__":
    crew = build_crew()
    result = crew.kickoff(inputs = {
            "text": "I love the new features in the latest update! "
                    "They make my experience so much better. "
                    "However, I found some bugs that need fixing. "
                    "Overall, it's a great improvement!"
        }
    )
    print(result)





