from stable_whisper import WhisperResult


def make_result(text):
    words = []
    start = 0.0
    for i, token in enumerate(text.split(' ')):
        word = token if i == 0 else f' {token}'
        words.append(dict(word=word, start=start, end=start + 0.1))
        start += 0.1
    return WhisperResult([dict(words=words)])


def _check_split_by_punctuation_min_chars_avoids_short_chunks():
    text = 'They spread ideas, behaviors, beliefs, and expressions in our culture through imitation.'
    result = make_result(text)

    result.split_by_punctuation([(',', ' '), '，'], min_chars=50)

    assert [segment.text for segment in result.segments] == [text]


def _check_split_by_punctuation_min_chars_uses_accumulated_chunk_length():
    text = 'This opening clause has enough words to pass the minimum character limit, short, tail'
    result = make_result(text)

    result.split_by_punctuation([(',', ' '), '，'], min_chars=50)

    assert [segment.text for segment in result.segments] == [
        'This opening clause has enough words to pass the minimum character limit,',
        ' short, tail'
    ]


def test():
    _check_split_by_punctuation_min_chars_avoids_short_chunks()
    _check_split_by_punctuation_min_chars_uses_accumulated_chunk_length()


if __name__ == '__main__':
    test()
