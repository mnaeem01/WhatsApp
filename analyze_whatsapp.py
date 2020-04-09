import calendar
import io
import logging
import os
import re
import sys
from datetime import datetime
import altair as alt


import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show, ion

import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

#color palette for ungal plot, needs more than 20 colors.
flatui = ["#000000", "#3498db", "#e74c3c", "#a3f441", "#34495e", "#2ecc71", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
          "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#e41a1c", "#8dd3c7", "#377eb8", "#999999", "#fbb4ae", "#b3cde3",
          "#ccebc5", "#decbe4", "#f781bf", "#ffff33", "#ff7f00", "#984ea3", "#4daf4a"]


user_continent_dict = {
    "Asim Naveed": "UAE",
    "Farrukh Shad": "Pakistan",
    "Ghyas Ud Deen": "UAE",
    "Jawaid": "Europe",
    "Kashif Aqeel": "North America",
    "Kazim Raza": "Pakistan",
    "Khuram Naeem": "North America",
    "Khurram Hamid": "North America",
    "Muhammad Owais": "Pakistan",
    "Munir Memon": "UAE",
    "Saad Rana": "Europe",
    "Shahzeb Sial": "North America",
    "Shajee Rafi": "UAE",
    "Tauseef Akhlaque": "North America",
    "Wajeeh": "UAE",
    "Zubair": "UAE",
    "Kamran Siddiqui": "Australia"
}


# file from Whatsapp export.
filepath = "data/_chat.txt"
content = ""

common_words = []
cw_filepath = "en_urdu_cw"
focus_month_start = '2020-02-01'
focus_month_end = '2020-04-01'
# new conversation starts after 15 minutes of inactivity.
time_between_conv_mins = 15
forwards = "video omitted|image omitted|GIF omitted|document omitted"

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
    return line.strip().replace("tauseef", "unda").replace("unday", "unda").replace("taufees","unda").replace("aanda","unda").replace("aanday","unda")

# remove non alphanumeric content
def get_words(msg):
    #if message is longer than 450 characters dont use it for cloud map. Its most probably a forward.
    if (len(msg)<450):
        regex = re.sub(r"[^a-z\s]+", "", msg.lower())
        regex = re.sub(r'[^\x00-\x7f]', r'', regex)
        #words = regex.split(" ")
    else:
        logging.debug("Length of msg %s", len(msg))
        logging.debug(msg)
        regex= "omitted"
    return regex


def convert_est_to_gst_tz(time_of_message_in_est):
    pst = pytz.timezone('America/New_York')
    time_of_message_in_est = pst.localize(time_of_message_in_est)
    #logging.debug("EST Time of Message: %s", time_of_message_in_est)
    # Convert to Dubai timezone
    dubai_pst = pytz.timezone('Asia/Dubai')
    time_of_message_in_gst = time_of_message_in_est.astimezone(dubai_pst)
    #logging.debug("GST Time of Message: %s", time_of_message_in_gst)
    return time_of_message_in_gst



logging.debug("Parsing csv file:%s", os.path.abspath(filepath))

try:
    with io.open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        content = replace_bad_characters(content)
except IOError as e:
    logging.error("Current Path %s", os.getcwd())
    logging.error("File %s not found. Please recheck your file location.", filepath)
    sys.exit()

if len(content) == 0:
    print("File is empty?")
    sys.exit()
#^\[([0-z \,\/]+?)\]\s+([0-z \+\-\(\)]+?):\s+(.*?)(?=^\[\d+\/\d+\/\d+\,)

#pattern = re.compile(r"""^\[(.*?)\]\s+(.*?):\s+(.*?)
#                            (?=^\[\d+\/\d+\/\d+\,)""", re.VERBOSE | re.DOTALL | re.MULTILINE)


pattern = re.compile(r"""^\[([0-z \,\/]+?)\]\s+([0-z\s\+‑\(\)]+?):\s+(.*?)
                            (?=^\[\d+\/\d+\/\d+\,)""", re.VERBOSE | re.DOTALL | re.MULTILINE)
list_of_messages = list()

for match in pattern.finditer(content):
    # Use this if US Eastern Time Zone to convert to dubai timezone.
    #[12/14/18, 12:00:27 PM] Shajee Rafi: https://www.dawn.com/news/1451019
    #time_of_message = (datetime.strptime(match.group(1), '%m/%d/%y, %I:%M:%S %p'))
    time_of_message = convert_est_to_gst_tz(datetime.strptime(match.group(1), '%m/%d/%y, %I:%M:%S %p'))


    # Some whatsapp exports send a different timestamp.
    # [5/22/18, 00:57:10] Afraz: <200e>image omitted^M"""
    # time_of_message = datetime.strptime(match.group(1), '%m/%d/%y, %H:%M:%S')

    sender = match.group(2)
    message = match.group(3).replace("\n", " ")
    list_of_messages.append([time_of_message, sender, message])

logging.debug("List of Messages %s", len(list_of_messages))

# create panda data frame from list of parsed messages
df_pd = pd.DataFrame(list_of_messages, columns=['date', 'user', 'message'])
df_pd['continent'] = df_pd['user'].map(user_continent_dict)
#check only focus month.
df_pd = df_pd[(df_pd['date'] > focus_month_start) & (df_pd['date'] < focus_month_end)]

#Add columns for analysis.
df_pd['reply_time'] = (df_pd.date.shift(-1) - df_pd.date).apply(lambda x: x.total_seconds() / 60).fillna(np.inf)
df_pd['conversation'] = (df_pd.reply_time > time_between_conv_mins).cumsum().shift(1).fillna(0).astype(int) + 1
df_pd['fwd_messages'] = (df_pd['message'].str.contains(forwards).fillna(0).astype(int))
df_pd['actual_messages'] = ((~df_pd['message'].str.contains(forwards)).fillna(0).astype(int))


logging.debug(df_pd)
df_group_by_conv = df_pd.groupby('conversation').agg({'date': ['min', 'max', 'count'],
                                                      'user': ['first', 'unique', 'nunique']
                                                      })

#pd.set_option('display.max_colwidth', 64)

# Using ravel, and a string join, we can create better names for the columns:
df_group_by_conv.columns = ["_".join(x) for x in df_group_by_conv.columns.ravel()]
df_group_by_conv['duration'] = (df_group_by_conv['date_max'] - df_group_by_conv['date_min']).apply(
    lambda x: int(x.total_seconds() / 60))

logging.debug(df_group_by_conv.head(20))
# date_count = number of messages in the conversation
df_group_by_conv = (df_group_by_conv[(df_group_by_conv['date_count'] > 10)]).sort_values('date_count', ascending=False)
logging.debug(df_group_by_conv)
df_group_by_conv.to_html('data/filename.html')

#group by conv initiator and people involved in conv.

df_group_by_conv_people = (df_group_by_conv.drop('user_unique', axis=1).join
                         (
                         df_group_by_conv.user_unique
                         .apply(pd.Series)
                         .stack()
                         .reset_index(drop=True, level=1)
                         .rename('user_unique')
                         ))
df_group_by_conv_people = df_group_by_conv_people[['user_first', 'user_unique']].reset_index()
logging.debug(df_group_by_conv_people)
df_group_by_conv_people= (df_group_by_conv_people.groupby(['user_first','user_unique'])
                                                .size()
                                                .to_frame('count')
                                                .reset_index())
df_group_by_conv_people = df_group_by_conv_people[df_group_by_conv_people['user_first']
                                                  != df_group_by_conv_people['user_unique']]
logging.debug(df_group_by_conv_people)
df_group_by_conv_people= df_group_by_conv_people.pivot('user_first','user_unique','count')
logging.debug(df_group_by_conv_people)

if not df_group_by_conv_people.empty:
    sns.set()
    sns.set_palette(sns.color_palette(flatui))
    #sns.palplot(sns.color_palette(flatui))

    group_by_conv_people_plot = df_group_by_conv_people.plot(kind='barh', stacked=True)
    #for i, v in enumerate(df_group_by_conv_people["user_nunique"]):
     #   group_by_conv_people_plot.text(i - .15, v + .15, v, color="black")


    plt.title('Ungals')
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.ylabel('Ungal Starter', fontsize=14)
    #plt.legend()
    draw()


df_group_by_conv_unique_user = (df_group_by_conv.groupby('user_first').count()[['user_nunique']])
# df4_pd.set_index(['user_first'],inplace=True)

df_group_by_conv_unique_user.sort_values('user_nunique', ascending=False, inplace=True)
logging.debug(df_group_by_conv_unique_user)


if not df_group_by_conv_unique_user.empty:
    sns.set()
    member_plot = df_group_by_conv_unique_user.plot(kind='bar', facecolor='lightblue')
    for i, v in enumerate(df_group_by_conv_unique_user["user_nunique"]):
        member_plot.text(i - .15, v + .15, v, color="black")
    plt.title('Ungals in Ramadan')
    plt.legend()
    draw()

group_by_day = df_pd.groupby(df_pd['date'].map(lambda x: x.date())).count()[['message']]


#not needed since only a single month.
# group_by_month = df_pd.groupby(df_pd['date'].dt.to_period('M')).count()[['message']]
# logging.debug(group_by_month.head(20))
# if not group_by_month.empty:
#     sns.set()
#     group_by_month_plot = group_by_month.plot(kind='bar', facecolor='red')
#     for i, v in enumerate(group_by_month["message"]):
#         group_by_month_plot.text(i - .15, v + .15, v, color="black")
#     plt.show()

#Group by user
logging.debug("group_by_day")
logging.debug(group_by_day)

group_by_day_user = df_pd.groupby([df_pd['date'].dt.to_period('D').astype(str), df_pd['user']]).count()[['message']].reset_index()
logging.debug("group_by_day_user")
logging.info(group_by_day_user)
group_by_user_by_day = pd.pivot_table(group_by_day_user, values='message', index=['user'], columns=['date'], fill_value=0).cumsum(axis=1)
logging.debug(group_by_user_by_day)

group_by_user_by_day.to_csv('data/group_by_user.csv')

#Group by continent
group_by_day_cont = df_pd.groupby([df_pd['date'].dt.to_period('D'), df_pd['continent']]).count()[['message']].reset_index()
group_by_cont_by_day = pd.pivot_table(group_by_day_cont, values='message', index=['continent'], columns=['date'], fill_value=0).cumsum(axis=1)
logging.debug(group_by_cont_by_day)

group_by_cont_by_day.to_csv('data/group_by_cont.csv')


"""
Altair Graph Group by User 
"""
#group_by_day_user.date = group_by_day_user.date.tz_localize("utc").tz_convert("Europe/Berlin").astype(str)
chart = alt.Chart(group_by_day_user).mark_circle(
    opacity=0.8,
    stroke='black',
    strokeWidth=1
).encode(
    alt.X('date:O', axis=alt.Axis(labelAngle=0)),
    alt.Y('user:N'),
    alt.Size('message:Q',
        scale=alt.Scale(range=[0, 5000]),
        legend=alt.Legend(title='Daily Messages')
    ),
    alt.Color('user:N', legend=None)
).properties(
    width=960,
    height=700
).transform_filter(
    alt.datum.user != '+971 50 527 3592'
).interactive()
chart.save('data/chart.html')

"""
Plot Daily Chat counts.
"""
if not group_by_day.empty:
    sns.set()
    group_by_day_plot = group_by_day.plot(kind='bar', facecolor='red')
    for i, v in enumerate(group_by_day["message"]):
        group_by_day_plot.text(i - .15, v + .15, v, color="black")
    draw()

"""
Plot User counts.
"""
group_by_user = df_pd.groupby(df_pd['user']).sum() \
    [['actual_messages', 'fwd_messages']].sort_values('actual_messages', ascending=False).head(18)
logging.debug("\ngroup_by_user")
logging.debug(group_by_user)



if not group_by_user.empty:
    sns.set()
    group_by_user_plot = group_by_user.plot(kind='bar', stacked=True)

    for i, (v_fwd, v_act) in enumerate(zip(group_by_user["fwd_messages"], group_by_user["actual_messages"])):
        v_total = v_fwd + v_act
        # if (v_fwd > 5):
        group_by_user_plot.text(i - .15, v_total + 10, v_fwd)
        group_by_user_plot.text(i - .15, v_act - 15, v_act, rotation=90)
    draw()

group_by_user_fwds_df = group_by_user
group_by_user_fwds_df['fwds_ratio'] = group_by_user.apply(lambda x: int((x['fwd_messages']/(x['fwd_messages'] + x['actual_messages']))*100), axis=1)
group_by_user_fwds_df = group_by_user_fwds_df.sort_values(['fwds_ratio'], ascending=False).reset_index()
logging.debug(group_by_user_fwds_df)
"""
Plot User forwards.
"""
if not group_by_user_fwds_df.empty:
    sns.set()
    group_by_user_fwds_plot = group_by_user_fwds_df.plot(x='user', y='fwds_ratio', kind='bar', stacked=False)

    for i, v_fwd_ratio in enumerate(group_by_user_fwds_df["fwds_ratio"]):
        group_by_user_fwds_plot.text(i - .15, v_fwd_ratio, str(v_fwd_ratio)+"%")
    draw()


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
    draw()
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
draw()


# df_pd.set_index('user').message.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'message'})
# print(df_pd)
word_count_by_user_df = df_pd[['user', 'message']][df_pd['message'].str.len() < 450 ]

word_count_by_user_df = (word_count_by_user_df.drop('message', axis=1).join
                         (
                         word_count_by_user_df.message
                         .apply(lambda x: re.sub(r"[^a-z\s]+", "", x.lower()))
                         .apply(lambda x: ' '.join([w for w in x.split() if w not in common_words]))
                         .apply(lambda x: replace_names(x))
                         .str.split(expand=True)
                         .stack()
                         .reset_index(drop=True, level=1)
                         .rename('message')
                         ))

word_count_by_user_df = word_count_by_user_df.dropna()

#df['count'] = 1

#df= df.groupby(['user', 'message']).sum()['count'].reset_index()
#print(df.sort_values(by=['user', 'count'], ascending=False).head(2))



word_count_by_user_df = (word_count_by_user_df.groupby(['user', 'message']).size().to_frame('count'))
# print(df.info())
# df=df.reset_index()
# print(df.info())
#print(df.groupby('user')['count'].nlargest(5))
word_count_by_user_df=(word_count_by_user_df.sort_values(['user', 'count'], ascending=False).groupby('user').head(5).reset_index())
word_count_by_user_df=(word_count_by_user_df[word_count_by_user_df['count'] > 5])
logging.debug(word_count_by_user_df)
if not word_count_by_user_df.empty:
    sns.set()
    group_by_user_wc_plot = word_count_by_user_df.plot.barh(x='user', y='count', stacked=False)

    for i, (m, c) in enumerate(zip(word_count_by_user_df["message"], word_count_by_user_df["count"])):
        # if (v_fwd > 5):
        group_by_user_wc_plot.text(c, i - .15, m)

    plt.show()
