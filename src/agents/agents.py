from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

def build_search_agent():

    return create_agent(
        model=llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )



#writer chain

# ------------------------------
# Writer Chain
# ------------------------------
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured, and insightful reports."),
    ("human", """Write a detailed research report on the topic below:

Topic = {topic}

Research Gathered = {research}

Structure the report as follows:
  1. Introduction
  2. Key Findings (minimum 3 well-explained findings)
  3. Conclusion
  4. Sources (list all URLs used in the research)

Be detailed, factual, and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ------------------------------
# Critic Chain
# ------------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical research reviewer. Evaluate reports for clarity, accuracy, completeness, and structure."),
    ("human", """Review the following research report and provide actionable feedback, pointing out strengths and areas for improvement:

Report:
{report}"""),
])

critic_chain = critic_prompt | llm | StrOutputParser()