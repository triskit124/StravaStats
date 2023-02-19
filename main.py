import os
import re

import pandas as pd
import numpy as np
import plotly

import matplotlib.pyplot as plt

from fitdecode import FitReader

DATA_DIR = "data"


"""
IDEAS

- time of day breakdown
- hours after sunset / before sunrise
- wordcloud for titles / descriptions
- distance
- time
- elevation
- day of week
- distance to elevation ratio (or similar metric)
- wind / temp

- weather / sun / uv vs percieved exertion

"""


def activity_name_analysis(activities):
    """
    Activity Name Analysis.
    """

    word_counts = {}
    names = " ".join(activities["Activity Name"].to_list())  # create one giant string
    names = re.sub(r"[^\w\s]", "", names)  # regex magic to remove punctation
    word_list = names.lower().split(" ")

    # count occurences of unique words
    for word in word_list:
        # ignore boring/default words
        if re.match(
            "a|the|an|the|to|in|for|of|or|by|with|is|on|that|be|afternoon|morning|evening|lunch",
            word,
        ):
            continue
        if word not in word_counts.keys():
            word_counts[word] = 0
        word_counts[word] += 1

    # sort based on word frequency
    word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1]))

    # set of unique words
    unique_words = set(word_counts.keys())

    return unique_words, word_counts


def make_word_cloud(
    word_counts: dict[str, int], filename: str, show_image: bool = True
) -> None:
    """
    Generates a word cloud from a word counts dictionary
    """
    from wordcloud import WordCloud
    from PIL import Image

    # make WordCloud
    image_mask = np.array(Image.open(DATA_DIR + "/bike_mask_2.png"))
    wordcloud = WordCloud(
        background_color="black",
        width=400,
        height=300,
        normalize_plurals=False,
        include_numbers=True,
        colormap="magma",
        repeat=True,
        prefer_horizontal=0.8,
        scale=1,
        min_font_size=4,
        margin=10,
        max_words=5000,
        mask=image_mask,
    )
    wordcloud.generate_from_frequencies(word_counts)
    wordcloud.to_file(filename)

    # plot the WordCloud image
    if show_image:
        plt.figure(facecolor=None)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()


if __name__ == "__main__":

    activities = pd.read_csv(DATA_DIR + "/activities.csv")

    # word analysis
    unique_activity_name_words, activity_name_word_counts = activity_name_analysis(
        activities
    )
    make_word_cloud(
        word_counts=activity_name_word_counts,
        filename=DATA_DIR + "/wordcloud.png",
        show_image=False,
    )
