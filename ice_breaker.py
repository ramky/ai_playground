from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile

DEBUG = False


def ice_break(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    if DEBUG:
        print(f"linkedin_profile_url is {linkedin_profile_url}")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    if DEBUG:
        print(f"linkedin_data is {linkedin_data}")

    print(chain.run(information=linkedin_data))


if __name__ == "__main__":
    # ice_break(name="Vidyut Latay")
    ice_break(name="Harrison Chase")
