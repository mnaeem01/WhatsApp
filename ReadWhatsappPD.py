import pandas as pd
import re


def parse_file(text_file):
    '''Convert WhatsApp chat log text file to a Pandas dataframe.'''

    # some regex to account for messages taking up multiple lines
    pat = re.compile(r'^(\[\d+\/\d+\/\d+\,.*?)(?=^\[\d+\/\d+\/\d+|\Z)', re.S | re.M)
    with open(text_file) as f:
        data = [m.group(1).strip().replace('\n', ' ').replace(u"\u202a", "").replace(u"\u200e", "").replace(u"\u202c", "") for m in pat.finditer(f.read())]

    sender = []
    message = []
    datetime = []
    for row in data:
        print(row)
        # timestamp is before the first dash
        datetime.append(row.split(' - ')[0])

        # sender is between am/pm, dash and colon
        try:
            s = re.search('m - (.*?):', row).group(1)
            sender.append(s)
        except:
            sender.append('')

        # message content is after the first colon
        try:
            message.append(row.split(': ', 1)[1])
        except:
            message.append('')

    df = pd.DataFrame(zip(datetime, sender, message), columns=['timestamp', 'sender', 'message'])
    df['timestamp'] = pd.to_datetime(df.timestamp, format='%d/%m/%Y, %I:%M %p')

    # remove events not associated with a sender
    df = df[df.sender != ''].reset_index(drop=True)

    return df


df = parse_file('data/_chat.txt')
print(df.count())
