import calendar
import io
import logging
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

common_words = []
cw_filepath = "en_urdu_cw"
focus_month_start = '2019-03-01'
focus_month_end = '2019-04-01'
# new conversation starts after 15 minutes of inactivity.
time_between_conv_mins = 15

if len(cw_filepath) > 0:

    try:
        common_words = __import__(cw_filepath, globals(), locals(), [common_words]).common_words

    except:
        print("Error getting common word file location")
        sys.exit()
else:
    print("You skipped common word.")

#chat file has unnecessary unicode characters, remove them.
def replace_bad_characters(line):
    return line.strip().replace(u"\u202a", "").replace(u"\u200e", "").replace(u"\u202c", "").replace(u"\xa0", " ")

def replace_names(line):
    return line.strip().replace("tauseef", "unda").replace("unday", "unda")

# remove non alphanumeric content
def get_words(msg):
    regex = re.sub(r"[^a-z\s]+", "", msg.lower())
    regex = re.sub(r'[^\x00-\x7f]', r'', regex)
    #words = regex.split(" ")
    return regex


def convert_est_to_gst_tz(time_of_message_in_est):
    pst = pytz.timezone('America/New_York')
    time_of_message_in_est = pst.localize(time_of_message_in_est)
    logging.debug("EST Time of Message: %s", time_of_message_in_est)
    # Convert to Dubai timezone
    dubai_pst = pytz.timezone('Asia/Dubai')
    time_of_message_in_gst = time_of_message_in_est.astimezone(dubai_pst)
    logging.debug("GST Time of Message: %s", time_of_message_in_gst)
    return time_of_message_in_gst

# file from Whatsapp export.
filepath = "data/_chat.txt"
content = ""

logging.debug("Parsing csv file:%s", os.path.abspath(filepath))

try:
    with io.open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        content = replace_bad_characters(content)
except IOError as e:
    logging.error("Current Path %s", os.getcwd())
    logging.error("File %s not found. Please recheck your file location.", filepath)
    sys.exit()

pattern = re.compile(r"""^\[(.*?)\]\s+(.*?):\s+(.*?)
                            (?=^\[\d+\/\d+\/\d+\,)""", re.VERBOSE | re.DOTALL | re.MULTILINE)
list_of_messages = list()

for match in pattern.finditer(content):
    # Use this if US Eastern Time Zone to convert to dubai timezone.
    #[12/14/18, 12:00:27 PM] Shajee Rafi: https://www.dawn.com/news/1451019
    time_of_message = convert_est_to_gst_tz(datetime.strptime(match.group(1), '%m/%d/%y, %I:%M:%S %p'))

    # Some whatsapp exports send a different timestamp.
    # [5/22/18, 00:57:10] Afraz: <200e>image omitted^M"""
    # time_of_message = datetime.strptime(match.group(1), '%m/%d/%y, %H:%M:%S')

    sender = match.group(2)
    message = match.group(3).replace("\n", " ")
    list_of_messages.append([time_of_message, sender, message])

logging.debug("List of Messages", len(list_of_messages))

# create panda data frame from list of parsed messages
df_pd = pd.DataFrame(list_of_messages, columns=['date', 'user', 'message'])
#check only focus month.
df_pd = df_pd[(df_pd['date'] > focus_month_start) & (df_pd['date'] < focus_month_end)]

#Add columns for analysis.
df_pd['reply_time'] = (df_pd.date.shift(-1) - df_pd.date).apply(lambda x: x.total_seconds() / 60).fillna(np.inf)
df_pd['conversation'] = (df_pd.reply_time > time_between_conv_mins).cumsum().shift(1).fillna(0).astype(int) + 1
df_pd['fwd_messages'] = (df_pd['message'].str.contains("omitted|http").fillna(0).astype(int))
df_pd['actual_messages'] = ((~df_pd['message'].str.contains("omitted|http")).fillna(0).astype(int))

logging.debug(df_pd)
df_group_by_conv = df_pd.groupby('conversation').agg({'date': ['min', 'max', 'count'],
                                                      'user': ['first', 'unique', 'nunique']
                                                      })

pd.set_option('display.max_colwidth', 64)

# Using ravel, and a string join, we can create better names for the columns:
df_group_by_conv.columns = ["_".join(x) for x in df_group_by_conv.columns.ravel()]
df_group_by_conv['duration'] = (df_group_by_conv['date_max'] - df_group_by_conv['date_min']).apply(
    lambda x: int(x.total_seconds() / 60))

logging.debug(df_group_by_conv.head(20))
# date_count = number of messages in the conversation
df_group_by_conv = (df_group_by_conv[(df_group_by_conv['date_count'] > 20)]).sort_values('date_count', ascending=False)
logging.debug(df_group_by_conv)
df_group_by_conv.to_html('data/filename.html')

df_group_by_conv_unique_user = (df_group_by_conv.groupby('user_first').count()[['user_nunique']])
# df4_pd.set_index(['user_first'],inplace=True)

logging.debug(df_group_by_conv_unique_user.sort_values('user_nunique', ascending=False, inplace=True))

if not df_group_by_conv_unique_user.empty:
    sns.set()
    member_plot = df_group_by_conv_unique_user.plot(kind='bar', facecolor='lightblue')
    for i, v in enumerate(df_group_by_conv_unique_user["user_nunique"]):
        member_plot.text(i - .15, v + .15, v, color="black")
    plt.title('Ungals in March')
    plt.legend()
    plt.show()

group_by_day = df_pd.groupby(df_pd['date'].map(lambda x: x.date())).count()[['message']]


#not needed since only a single month.
#group_by_month = df_pd.groupby(df_pd['date'].dt.to_period('M')).count()[['message']]
#logging.debug(group_by_month.head(20))
# if not group_by_month.empty:
#     sns.set()
#     group_by_month_plot = group_by_month.plot(kind='bar', facecolor='red')
#     for i, v in enumerate(group_by_month["message"]):
#         group_by_month_plot.text(i - .15, v + .15, v, color="black")
#     plt.show()

"""
Plot Daily Chat counts.
"""
if not group_by_day.empty:
    sns.set()
    group_by_day_plot = group_by_day.plot(kind='bar', facecolor='red')
    for i, v in enumerate(group_by_day["message"]):
        group_by_day_plot.text(i - .15, v + .15, v, color="black")
    plt.show()

"""
Plot User counts.
"""
group_by_user = df_pd.groupby(df_pd['user']).sum() \
    [['actual_messages', 'fwd_messages']].sort_values('actual_messages', ascending=False).head(15)

if not group_by_user.empty:
    sns.set()
    group_by_user_plot = group_by_user.plot(kind='bar', stacked=True)

    for i, (v_fwd, v_act) in enumerate(zip(group_by_user["fwd_messages"], group_by_user["actual_messages"])):
        v_total = v_fwd + v_act
        # if (v_fwd > 5):
        group_by_user_plot.text(i - .15, v_total + 10, v_fwd)
        group_by_user_plot.text(i - .15, v_act - 15, v_act, rotation=90)
    plt.show()

lambdafunc = lambda x: pd.Series([calendar.day_name[x['date'].weekday()], x['date'].hour])

df_pd[['weekday', 'hour']] = df_pd.apply(lambdafunc, axis=1)

"""
Heatmap
"""
heatmap_df = df_pd
logging.debug(heatmap_df)

if not heatmap_df.empty:
    grouped_heatmap = heatmap_df.groupby(["weekday", "hour"]).count()[['message']]
    pivoted_heatmap = pd.pivot_table(grouped_heatmap, values='message', index=['weekday'], columns=['hour'])
    plt.figure(figsize=(5, 16))
    sns.heatmap(pivoted_heatmap,
                annot=True,
                fmt=".0f",
                linewidths=.2,
                cmap="YlGnBu",
                cbar=False
                )

    plt.show()
else:
    logging.debug("This chat does not contain any datetime")

"""
Word Cloud. 
"""
message_text = " ".join(m for m in df_pd['message'].apply(get_words))

message_text= replace_names(message_text)

logging.debug("There are {} words in the combination of all messages.".format(len(message_text)))

wordcloud = WordCloud(
    width=500,
    height=1000,
    background_color="white",
    stopwords=common_words
).generate(message_text)

plt.figure(figsize=(8, 16))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
