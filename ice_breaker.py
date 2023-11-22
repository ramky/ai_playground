from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from output_parsers import person_intel_parser, PersonIntel
from tools.others.linkedin_data_stub import linkedin_data_stub

DEBUG = False
SIMULATE = True


def ice_break(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    if DEBUG:
        print(f"linkedin_profile_url is {linkedin_profile_url}")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. A topic that may interest them
        Format instructions:
        {format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    if SIMULATE:
        linkedin_data = scrape_linkedin_profile(
            linkedin_profile_url=linkedin_profile_url
        )
        if DEBUG:
            print(f"linkedin_data is {linkedin_data}")
    else:
        linkedin_data = linkedin_data_stub

    result = chain.run(information=linkedin_data)

    return person_intel_parser.parse(result)


if __name__ == "__main__":
    # result = ice_break(name="Vidyut Latay")
    result = ice_break(name="Harrison Chase")
    print(f"result is {result}")
