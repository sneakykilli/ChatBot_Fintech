from bow import prediction
import pandas as pd
from preprocessing import df
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

EXIT_KEYWORDS = [
    "quit", "exit", "goodbye", "bye", "stop", "end", "cancel", "close",
    "finish", "done", "no more", "i'm done", "thanks, that's all",
    "that's it", "wrap up", "terminate", "abandon", "cease", "halt", "sign off", 'no']
POSITIVE_EXITS = ['yes', 'please']
WELCOME = """
Welcome to KilliBank , I'm here to help. 
At anytime you can end this chat by typing 'End'
Or speak to an Agent by typing 'Agent'
"""
HELP = """
How can I help?
"""
HELP_AGAIN = """
Is there anything else I can help you with?
"""
STOP_CONFIRMATION = """
Type 'Agent' to Talk to an agent 
or 
'End' to end chat. 
"""
GOODBYE = """
Thank you for chatting, have a Great day!
"""

helper = True
print(WELCOME)
user_response = input(HELP).lower()
while helper:
    if user_response not in EXIT_KEYWORDS:
        guess = prediction(user_response)
        response_df = df[(df['Topic'] == guess[1]) & (df['Action'] == guess[2])]
        response = response_df['Resolution'].to_string(index=False)
        response = response.replace('\\n', '\n')
        print(response)
        user_response = input(HELP_AGAIN).lower()
    else:
        user_response = input(STOP_CONFIRMATION).lower()
        if user_response.lower() in EXIT_KEYWORDS or user_response.lower() in POSITIVE_EXITS:
            print(GOODBYE)
            helper = False