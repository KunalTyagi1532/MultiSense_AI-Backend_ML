import emoji


class EmojiProcessor:

    @staticmethod
    def process(text):

        return emoji.demojize(text)