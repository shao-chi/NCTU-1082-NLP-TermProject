import re
import emoji

import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

from emosent import get_emoji_sentiment_rank
from redditscore.tokenizer import CrazyTokenizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class PreprocessTweet:
    def __init__(self, path):
        self.__dataframe = pd.read_json(path, lines=True)
        self.__dataframe['preprocessed_text'] = self.__dataframe['text'] \
            .str.cat(self.__dataframe['reply'].values, sep='\n')

        self.__dataframe['preprocessed_text'] = self.replace_new_lines()
        self.__dataframe['preprocessed_text'] = self.replace_ampersand_character()
        self.__dataframe['preprocessed_text'] = self.replace_usertag()
        self.__dataframe['preprocessed_text'] = self.remove_URL()
        # self.__dataframe['preprocessed_text'] = self.replace_hashtag()
        self.__dataframe['preprocessed_text'] = self.replace_contraction()
        self.__dataframe['preprocessed_text'] = self.replace_emoticons()
        self.__dataframe['preprocessed_text'] = self.replace_emojis()
        self.__dataframe['preprocessed_text'] = self.replace_repeating_characters()
        # self.__dataframe['preprocessed_text'] = self.replace_punctation()
        self.__dataframe['preprocessed_text'] = self.replace_specific_characters()
        self.__dataframe['preprocessed_text'] = self.lemmatize_text()

        self.__dataframe['tokens'] = self.tokenize()
        self.__dataframe['tokens'] = self.remove_stop_words()

    def replace_new_lines(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: re.sub(r'\n', ' ', text))

    def replace_ampersand_character(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: text.replace('&amp;', ' and '))

    def replace_usertag(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: re.sub(r'@.*?( |$)', 'USERTAG ', text))

    def remove_URL(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: re.sub(r'http[s]{0,1}.*?( |$)', '', text))

    def replace_hashtag(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: text.replace('#', ''))

    def replace_contraction(self):
        def replace(text):
            # specific
            text = re.sub(r"won't", "will not", text)
            text = re.sub(r"can\'t", "can not", text)

            # general
            text = re.sub(r"n\'t", " not", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'s", " is", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'t", " not", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'m", " am", text)

            return text

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: replace(text))

    def replace_emoticons(self):
        def replace(text,
                    placeholder_pos=' POS_EMOTICON ',
                    placeholder_neg=' NEG_EMOTICON '):
            emoticons_pos = [":)", ":-)", ":p", ":-p", ":P", ":-P", ":D",":-D",
                             ":]", ":-]", ";)", ";-)", ";p", ";-p", ";P", ";-P",
                             ";D", ";-D", ";]", ";-]", "=)", "=-)", "<3"]
            emoticons_neg = [":o", ":-o", ":O", ":-O", ":(", ":-(", ":c", ":-c",
                             ":C", ":-C", ":[", ":-[", ":/", ":-/", ":\\", ":-\\",
                             ":n", ":-n", ":u", ":-u", "=(", "=-(", ":$", ":-$"]

            # replace positive emoticons by placeholder
            for e in emoticons_pos:
                text = text.replace(e, placeholder_pos)

            # replace negative emoticons by placeholder
            for e in emoticons_neg:
                text = text.replace(e, placeholder_neg)

            return text

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: replace(text))

    def replace_emojis(self):
        def replace(text,
                    placeholder_pos=' POS_EMOJI ',
                    placeholder_neg=' NEG_EMOJI ',
                    placeholder_net=' NETRAL_EMOI '):
            for str_ in text:
                if str_ in emoji.UNICODE_EMOJI:
                    try:
                        score = get_emoji_sentiment_rank(str_)['sentiment_score']
                    except KeyError:
                        # text = text.replace(str_, placeholder_net)
                        text = text.replace(str_, '')
                        continue

                    if score > 0:
                        text = text.replace(str_, placeholder_pos)
                    else:
                        text = text.replace(str_, placeholder_neg)

            return text

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: replace(text))

    def replace_repeating_characters(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: re.sub(r'(.)\1{2,}', r'\1\1', text))

    def replace_punctation(self):
        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: re.sub('[^\w\s]', ' ', text))

    def replace_specific_characters(self):
        def replace(text):
            text = text.replace(u'\u201c', ' ')	# double opening quotes
            text = text.replace(u'\u201d', ' ')	# double closing quotes
            text = text.replace(u'\u2014', ' ')	# -
            text = text.replace(u'\u2013', ' ') # -
            text = text.replace(u'\u2026', ' ') # horizontal elipsses ...

            return text

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: replace(text))

    def tokenize(self):
        word_tokenize = CrazyTokenizer(twitter_handles='split',
                                       hashtags='split',
                                       decontract=True)

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: word_tokenize.tokenize(text))

    def lemmatize_text(self):
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

        def lemmatize_words(text):
            pos_tagged_text = nltk.pos_tag(text.split())

            return " ".join([lemmatizer.lemmatize(
                                word, wordnet_map.get(pos[0], wordnet.NOUN)) \
                                for word, pos in pos_tagged_text])

        return self.__dataframe['preprocessed_text'] \
                .apply(lambda text: lemmatize_words(text))

    def remove_stop_words(self):
        stop_words = set(stopwords.words('english'))

        return self.__dataframe['tokens'] \
                .apply(lambda tokens: [w for w in tokens if w not in stop_words])
        
    def build_vocab(self):
        vocab = Counter()

        for tokens in self.__dataframe['tokens'].values:
            vocab.update(tokens)

        return vocab

    def add_pad(self, max_length):
        tokens_list = self.__dataframe['tokens'].values
        word_idx_list = self.__dataframe['word_index'].values

        # for i in range(len(tokens_list)):
        #     if len(tokens_list[i]) >= max_length:
        #         tokens_list[i] = ['<START>'] + tokens_list[i][:max_length]
        #         word_idx_list[i] = [1] + word_idx_list[i][:max_length]

        #     if len(tokens_list[i]) < max_length:
        #         add = max_length - len(tokens_list[i])
        #         pad = ['<PAD>' for _ in range(add)] + ['<START>']
        #         zero = [0 for _ in range(add)] + [1]

        #         tokens_list[i] = pad + tokens_list[i]
        #         word_idx_list[i] = zero + word_idx_list[i]

        for i in range(len(tokens_list)):
            if len(tokens_list[i]) >= max_length:
                word_idx_list[i] = word_idx_list[i][:max_length]

            if len(tokens_list[i]) < max_length:
                add = max_length - len(tokens_list[i])
                pad = ['<PAD>' for _ in range(add)]
                zero = [0 for _ in range(add)]

                tokens_list[i] = pad + tokens_list[i]
                word_idx_list[i] = zero + word_idx_list[i]

        self.__dataframe['tokens'] = pd.Series(tokens_list)
        self.__dataframe['word_index'] = pd.Series(word_idx_list)

    def word_to_index(self, to_index):
        self.__dataframe['tokens'] = self.__dataframe['tokens'] \
                .apply(lambda tokens:
                    [w for w in tokens if w in to_index.keys()])

        self.__dataframe['preprocessed_text'] = self.__dataframe['tokens'] \
                .apply(lambda tokens: ' '.join(tokens))

        self.__dataframe['word_index'] = self.__dataframe['tokens'] \
                .apply(lambda tokens:
                    [to_index[w] for w in tokens])

    @property
    def dataframe(self):
        return self.__dataframe
