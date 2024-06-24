import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

DEBUG_MODE = True

load_dotenv()


@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    scaler = MinMaxScaler()
    # Normalize data excluding identifier columns and 'name of car'
    data_scaled = scaler.fit_transform(data.drop(columns=["name of car"]))
    # Concatenate the 'name of car' column back to the scaled DataFrame
    return (
        pd.concat(
            [data["name of car"], pd.DataFrame(data_scaled, columns=data.columns[1:])],
            axis=1,
        ),
        scaler,
    )


def find_knn(input_data, data, scaler, n_neighbors=3):
    input_scaled = scaler.transform([input_data])  # Normalize input data
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data.iloc[:, 1:])  # Fit on normalized data excluding 'name of car'
    distances, indices = knn.kneighbors(input_scaled)
    return (
        data.iloc[indices[0]],
        distances,
    )  # Return the nearest neighbors including 'name of car'


class UserScore(BaseModel):
    risk_profile_score: Optional[float] = Field(
        None,
        description="Score for risk preference (1 - 10 integer, higher indicates higher risk preference)",
    )
    social_profile_score: Optional[float] = Field(
        None,
        description="Score for socialness (1 - 10 integer, higher means the person is more social)",
    )
    curiosity_profile_score: Optional[float] = Field(
        None,
        description="Score for curiosity (1 - 10 integer, higher means more curious about nature and the world)",
    )
    moral_profile_score: Optional[float] = Field(
        None,
        description="Score higher for if talking about strangers, close relatives, nature instead of personal achievement",
    )
    safety_profile_score: Optional[float] = Field(
        None, description="Score for safety features (1 - 10 integer)"
    )
    year_profile_score: Optional[float] = Field(
        None,
        description="Year of the song release",
    )
    one_liner: str = Field(
        description="Use something in the lyrics of the user's favorite song to generate a funny one liner about the car, some kind of pun."
    )


def get_year_score(year: int) -> int:
    if 1960 <= year < 1970:
        return 1
    elif 1970 <= year < 1980:
        return 2
    elif 1980 <= year < 1990:
        return 3
    elif 1990 <= year < 1995:
        return 4
    elif 1995 <= year < 2000:
        return 5
    elif 2000 <= year < 2005:
        return 6
    elif 2005 <= year < 2010:
        return 7
    elif 2010 <= year < 2015:
        return 8
    elif 2015 <= year < 2020:
        return 9
    elif 2020 <= year < 2025:
        return 10
    else:
        return 0


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
user_scorer = model.with_structured_output(UserScore)

st.set_page_config(page_title="Toyota Finder")

st.title("Toyota Finder")
st.header("Find your perfect toyota")

# Load data and get scaler
data, scaler = load_data("data.csv")

# Creating a form to hold our questions and submit button
with st.form(key="car_preference_form"):
    # Each question corresponds to a metric in your UserScore model
    risk_profile = st.text_area(
        "Describe the riskiest endeavour you've done in your life?"
    )
    social_profile = st.text_area("Who inspires you most?")
    curiosity_profile = st.text_area(
        "If you could learn about any topic in the world, no matter how obscure or unrelated to your current knowledge or profession, what would it be and why?"
    )
    moral_profile = st.text_area("When did you feel the most amount of joy?")
    safety_profile = st.text_area("Where do you feel safest?")
    year_profile = st.text_area("What is your favorite song?")

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # Override with test data if debugging
    if DEBUG_MODE:
        risk_profile = "Not putting sunscreen"
        social_profile = "I don't like partying"
        curiosity_profile = "I would learn about puppies because they are cute"
        moral_profile = "I feel most amount of joy when i give everything i've ever worked for to people I dont know"
        safety_profile = "I feel safest when I stare at a sunset"
        year_profile = "Michael jackson beat it"

    user_input = f"""
Q1: Describe the riskiest endeavour you've done in your life? [Risk Score]
(if actually risky - higher score)
A1: {risk_profile}
E1: [
    Not putting sunscreen - 1
    cheating on an exam - 2
    Something illegal - 8
    Throwing yourself in front of a train to save someones life - 10
]


Q2: Describe yourself at a party [Social Score]
(if alone - low social)
(talk to strangers - very high score)
(talk to friends - medium high score)
A2: {social_profile}
E2: [
    I dont like partying - 1
    I like to be in the corner and wait for others to talk to me - 2
    I will look for my friends - 5
    I will talk to every stranger at the party and introduce myself - 10
]


Q3: If you could learn about any topic in the world, no matter how obscure or unrelated to your current knowledge or profession, what would it be and why?" [Curiosity score]
(if its a topic that relates to nature - higher score)
(if describe a complex topic - higher score)
(if genuine interest shown - higher score)
Else (lower score)
A3: {curiosity_profile}
E3: [
    I wouldn't learn, learning is boring - 1
    I would learn about puppies because they are cute - 5
    I would love to learn about mathematics because I want to discover how the world works - 9
    Oceanography captivates me because it unveils the mysteries of our planet's most enigmatic and dynamic ecosystem: the ocean. The thrill of exploring uncharted depths, discovering new species, and witnessing the incredible diversity of marine life is unparalleled. The ocean is a world of constant motion and change, where powerful currents, awe-inspiring underwater landscapes, and vibrant coral reefs create a mesmerizing tapestry of life. The study of oceanography not only feeds my curiosity about the natural world but also holds the key to understanding crucial environmental processes and addressing global challenges like climate change. Every dive, every expedition, and every research project deepens my appreciation for the ocean's profound beauty and its vital importance to life on Earth. This passion fuels my commitment to preserving and protecting this vast, blue frontier for future generations. - 10
]


Q4: When did you feel the most amount of joy [moral rank]
(if talks about strangers - very high score)
(if talks about close relatives (high score)
(if talks about nature - higher score)
(if talks about personal achievement then lower score)
A4: {moral_profile}
E4: [
    I feel the most joy when I go to the gym - 2
    I am joyful when I give to others - 7
    Joy comes from inner peace - 6
    I feel most amount of joy when i give everything i've ever worked for to people I dont know - 10
]


Q5: Where do you feel safest? [safety rank, lower is less safe]
(abstract concept - low score)
(concrete concept - medium score)
(physical space - high score)
A5: {safety_profile}
E5: [
    I feel safest when I stare at a sunset - 2
    I feel safe in my house, where it is warm and with my family - 7
    I feel safe in a nuclear bunker - 10
]

Q6: What is your favourite song? [year]
A6: {year_profile}
"""
    st.success("Finding your perfect toyota...")
    response: UserScore = user_scorer.invoke(
        f"You are an expert car recommender. Give an accurate score for the user's preferences based on the following user input: {user_input}"
    )

    st.warning(response)

    user_scores = [
        response.risk_profile_score,
        response.social_profile_score,
        response.curiosity_profile_score,
        response.moral_profile_score,
        response.safety_profile_score,
        get_year_score(response.year_profile_score),
    ]

    # Now including scaler in the kNN search
    neighbors, distances = find_knn(user_scores, data, scaler, n_neighbors=3)

    st.write(response.one_liner)
    st.write(distances)
    st.write(neighbors)
