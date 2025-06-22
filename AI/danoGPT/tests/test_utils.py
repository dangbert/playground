import string

import utils


def test_encoder_decoder():
    chars = [c for c in string.ascii_letters + " "]
    encode, decode = utils.get_encoder_decoder(chars)
    assert decode(encode("hello world")) == "hello world"
