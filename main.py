import os
import re
import math

import pandas as pd
import numpy as np
import plotly

import matplotlib.pyplot as plt

from fitdecode import FitReader


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


def parse_gpx_file(filename: str) -> pd.DataFrame:
    """
    Reads a .gpx file and returns a Pandas DataFrame of:
        - timestamp
        - lat, lon, h
        - ECEF coordinates
        - grade
    """

    import xml.etree.ElementTree as ET
    from utils import llh_to_ecef, ecef_to_enu

    tree = ET.parse(filename)
    root = tree.getroot()

    # for some reason all tags are prefixed with something of the form: '{http://www.topografix.com/GPX/1/1}'
    prefix = root.tag.replace('gpx', '')

    # intialize values
    lat, lon, ele, time, x, y, z, grade = [], [], [], [], [], [], [], []

    # loop through each track point and get lat, lon, ele, time, grade, etc
    for i, pt in enumerate(root.iter(prefix + 'trkpt')):
        lat.append(float(pt.attrib['lat']))
        lon.append(float(pt.attrib['lon']))

        for child in pt:
            if child.tag == prefix + 'ele':
                ele.append(float(child.text))
            elif  child.tag == prefix + 'time':
                time.append(child.text)
            # else:
            #     assert 0, "Unkown child tag: {} in gpx file {}".format(child.tag, filename)

        # calculate x, y, z of current point from lat, lon, height
        x_ECEF, y_ECEF, z_ECEF = llh_to_ecef(lat[-1], lon[-1], ele[-1])
        
        x.append(x_ECEF)
        y.append(y_ECEF)
        z.append(z_ECEF)

        # compute grade based on ENU location wrt previous point
        if i == 0:
            grade.append(0)
        else:
            # ENU coordinates of current point, w.r.t local tangent plane centered at previous point
            east, north, up = ecef_to_enu(x_ECEF, y_ECEF, z_ECEF, lat[-2], lon[-2], ele[-2])
            r = math.sqrt(east**2 + north**2)
            if r > 0:
                grade.append(100 * up / r)
            else:
                grade.append(0)

        assert len(lat) == len(lon) == len(ele) == len(time), "Missing data in gpx file " + filename

    d = {
        'timestamp': time,
        'latitude [deg]': lat,
        'longitude [deg]': lon,
        'elevation [m]': ele,
        'x (ECEF) [m]': x,
        'y (ECEF) [m]': y,
        'z (ECEF) [m]': z,
        'grade [%]': grade,
    }

    return pd.DataFrame(d)

def parse_activity_files(activities: pd.DataFrame, data_dir: str, save_to_csv: bool = True):
    """
    Parses all activity files and optionally saves them as .csv files. Calculates additional information such as grade.
    """

    for i, activity_file in enumerate(activities['Filename']):

        # ignore empty entries
        if type(activity_file) != str:
            continue
        
        filename = data_dir + "/" + activity_file.replace(".gz", "")

        # parse file according to its data format
        if '.gpx' in filename:
            data = parse_gpx_file(filename)
        elif '.fit' in filename:
            continue
        else:
            raise NotImplementedError

        print('Processed ' + filename + " ...")

        if save_to_csv:
            data.to_csv(filename + '.csv')

        # fill in missing grade information into activities dataframe
        grade = data['grade [%]']
        ele = data['elevation [m]']

        avg_positive_grade = np.mean(grade[grade > 1e-1])
        avg_negative_grade = np.mean(grade[grade < -1e-1])

        activities.at[i, "Average Positive Grade"] = avg_positive_grade
        activities.at[i, "Average Negative Grade"] = avg_negative_grade

    if save_to_csv:
        activities.to_csv(data_dir + "/activities_PROCESSED.csv")


def generate_heatmap(data_dir: str):

    import plotly.express as px
    import plotly.graph_objects as go

    csv_files = [file for file in os.listdir(data_dir + '/activities/') if '.csv' in file]
    # csv_files = csv_files[0:100]
    dataframes = []

    fig = go.Figure()

    for file in csv_files:
        data = pd.read_csv(data_dir + "/activities/" + file)
        dataframes.append(data)

        fig.add_trace(
            go.Scattergeo(
                locationmode = 'USA-states',
                lon = data['longitude [deg]'],
                lat = data['latitude [deg]'],
                mode = 'lines',
                line = dict(width = 2, color = 'red'),
                opacity = 0.2,
            )
        )

    all_data = pd.concat(dataframes)

    fig.update_layout(
        title_text = 'heatmap.',
        showlegend = False,
        geo = dict(
            scope = 'north america',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    )

    # fig = px.density_mapbox(all_data, lat='latitude [deg]', lon='longitude [deg]', z='elevation [m]', radius=1,
    #                     center=dict(lat=37, lon=-121), zoom=0,
    #                     mapbox_style="stamen-terrain")

    # fig = px.line_mapbox(all_data, lat="latitude [deg]", lon="longitude [deg]", zoom=3, height=900)
    # fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=4, mapbox_center_lat = 37, margin={"r":0,"t":0,"l":0,"b":0})
    
    fig.show()


if __name__ == "__main__":

    WORD_ANALYSIS               = False
    WORD_CLOUD                  = False
    PARSE_ACTIVITY_FILES        = False
    GENERATE_HEATMAP            = True
    DATA_DIR                    = "data"

    activities = pd.read_csv(DATA_DIR + "/activities.csv")

    if WORD_ANALYSIS:
        unique_activity_name_words, activity_name_word_counts = activity_name_analysis(
            activities
        )
    
    if WORD_CLOUD:
        make_word_cloud(
            word_counts=activity_name_word_counts,
            filename=DATA_DIR + "/wordcloud.png",
            show_image=False,
        )

    if PARSE_ACTIVITY_FILES:
        parse_activity_files(activities=activities, data_dir=DATA_DIR, save_to_csv=True)

    if GENERATE_HEATMAP:
        generate_heatmap(DATA_DIR)

    
