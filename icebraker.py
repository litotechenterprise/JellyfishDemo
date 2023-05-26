from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv.main import load_dotenv
information = """
Hugh Marston Hefner (April 9, 1926 – September 27, 2017) was an American magazine publisher. He was the founder and editor-in-chief of Playboy magazine, a publication with revealing photographs and articles that provoked charges of obscenity.
Hefner extended the Playboy brand into a world network of Playboy Clubs. He also resided in luxury mansions where Playboy Playmates shared his wild partying life, fueling media interest. He was a political activist in the Democratic Party and 
for the causes of First Amendment rights, animal rescue, and the restoration of the Hollywood Sign. He was a highly controversial figure in popular culture, but he has been accused of perpetrating and fostering sexual abuse and exploitation 
stretching back decades, and Playboy has since distanced itself from association with him.

Hefner was born in Chicago on April 9, 1926,[4] the first child of Glenn Lucius Hefner (1896–1976), an accountant, and his wife Grace Caroline (Swanson) Hefner (1895–1997) who worked as a teacher. His parents were from Nebraska.
He had a younger brother, Keith (1929–2016).[7][8][9] His mother was of Swedish ancestry, and his father was German and English. Through his father's line, Hefner was a descendant of Plymouth governor William Bradford.He described his family 
as "conservative, Midwestern, [and] Methodist".[14] His mother had wanted him to become a missionary.He attended Sayre Elementary School and Steinmetz High School, then served from 1944 to 1946 as a U.S. Army writer for a military newspaper. 
Hefner graduated from the University of Illinois at Urbana–Champaign in 1949 with a Bachelor of Arts in psychology and a double minor in creative writing and art, having earned his degree in two and a half years. After graduation, he took a 
semester of graduate courses in sociology at Northwestern University, but dropped out soon after.
"""

if __name__ == "__main__":
    summary_template = """
        given the information {information} about a person from I want to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    print(chain.run(information=information))
1
