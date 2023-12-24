features_config = {
    "categorical": {
        "single": {
                   "language": {"vocab_size": 299},
                   "country": {"vocab_size": 222},
                   "start_year": {"vocab_size": 2050},
                   "crew": {"vocab_size": 186874},
                   "writer": {"vocab_size": 301708}
                  },
        "multi": {"genre": {"max_seq_len": 8, "vocab_size": 24+1},
                  "company": {"max_seq_len": 5, "vocab_size": 149046+1}, #22 149044 385443
                  "cast": {"max_seq_len": 3, "vocab_size": 385445+1}
                 }
    },
    "d_model": 16,
    "text": {"max_seq_len": 256, "num_encoder": 3, "d_model": 768},
    "revenue": {"scale": 100000000},
    "ff": {"dim": 1024, "dropout": 0.2}
}