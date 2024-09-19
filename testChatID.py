import requests
from telegram import Bot

TOKEN = '7270440999:AAGHWa8CdB3oL5kR3iwMsj0QgfSyi77RiXU'
url = f'https://api.telegram.org/bot{TOKEN}/getUpdates'
response = requests.get(url)
data = response.json()

if data['result']:
    for update in data['result']:
        chat_id = update['message']['chat']['id']
        print(f'Chat ID: {chat_id}')
else:
    print('No updates found. Please send a message to the bot and try again.')


# bot = Bot(token=TOKEN)

# updates = bot.get_updates()
# for update in updates:
#     print(update.message.chat_id)