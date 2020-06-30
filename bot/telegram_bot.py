import shutil

import telebot
from telebot.types import Message

import main

TOKEN = ''

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start_handler(message: Message):
    bot.send_message(message.chat.id, 'Hi. I am ObjectDetector bot.')


@bot.message_handler(commands=['help'])
def help_handler(message: Message):
    bot.send_message(message.chat.id, 'Just send me a pic.')


@bot.message_handler(content_types=['photo'])
def image_handler(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'bot/' + file_info.file_path
    det_src = 'bot/det_' + file_info.file_path
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    new_file.close()
    main.main()
    bot.send_photo(message.chat.id, open(det_src, 'rb'), 'Identified objects')
    shutil.move(src, 'photos')
    shutil.move(det_src, 'det_photos')


bot.polling()
