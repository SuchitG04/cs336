import copy
import multiprocessing as mp
import os
import regex as re

from collections import Counter, defaultdict
from typing import BinaryIO

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 10

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_word_count(
    corpus: str
) -> Counter:
    word_count = Counter()
    for word in re.finditer(PAT, corpus, re.IGNORECASE):
        word = word.captures()[0]
        byte_chars = tuple([bytes([o]) for o in list(word.encode("utf-8"))])
        word_count[byte_chars] += 1
    return word_count


def get_pair_stats(
    sorted_word_count: list[tuple[tuple, int]]
) -> tuple[Counter, defaultdict[tuple, Counter]]:
    pair_stats, pair_indices = Counter(), defaultdict(Counter)
    for i, (word, freq) in enumerate(sorted_word_count):
        for p in zip(word, word[1:]):
            pair_stats[p] += freq
            pair_indices[p][i] += 1
        
    return pair_stats, pair_indices


def pretokenize(
    f_start: int,
    f_end: int,
    f: BinaryIO,
    special_tok_pat: str,
    proc_q: mp.Queue
):

    f.seek(f_start)
    chunk = f.read(f_end - f_start).decode("utf-8", errors="ignore")

    word_count = Counter()
    for part in re.split(special_tok_pat, chunk):
        word_count.update(get_word_count(part))
    # pair_stats, pair_indices = get_pair_stats(word_count)
    
    # proc_q.put((word_count, pair_stats, pair_indices))
    proc_q.put(word_count)


def replace_pair(
    pair: tuple[bytes, bytes],
    sorted_word_count: list[tuple[tuple, int]],
    indices: defaultdict[tuple, Counter]
) -> list[tuple[int, tuple, tuple, int]]:

    changes = []

    for j, freq in indices[pair].items():
        if freq < 1:
            continue
        word, freq = sorted_word_count[j]
        first, second = pair
        new_tok = pair[0] + pair[1]

        i = 0
        new_word = ()
        while i+1 < len(word):
            if word[i] == first and word[i+1] == second:
                new_word += (new_tok, )
                i += 2
            else:
                new_word += (word[i], )
                i += 1
        if i < len(word): new_word += (word[i], )

        sorted_word_count[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))
    
    return changes


def update_pair_stats(
    pair: tuple[bytes, bytes],
    changes: list[tuple[int, tuple, tuple, int]],
    stats: Counter,
    indices: defaultdict[tuple, Counter]
):
    del stats[pair]
    del indices[pair]

    first, second = pair
    new_tok = first + second

    for j, word, old_word, freq in changes:

        i = 0
        while True:
            try:
                i = old_word.index(first, i)
            except ValueError:
                break

            if i+1 < len(old_word) and old_word[i+1] == second:
                # decrement left neighbour if present
                if i > 0:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1

                # decrement right neighbour
                if i+2 < len(old_word):
                    # walking on some thin indexing ice right here
                    # skip if the sequence is A B C B C and top_pair is B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        next_ = old_word[i+1:i+3]
                        stats[next_] -= freq
                        indices[next_][j] -= 1

                i += 2
            else:
                i += 1
        
        i = 0
        while True:
            try:
                i = word.index(new_tok, i)
            except ValueError:
                break
                
            if i > 0:
                prev = word[i-1:i+1]
                stats[prev] += freq
                indices[prev][j] += 1
            
            if i+1 < len(word) and word[i+1] != new_tok:
                next_ = word[i:i+2]
                stats[next_] += freq
                indices[next_][j] += 1
            i += 1


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    escaped_tokens = [re.escape(o) for o in special_tokens]
    special_tok_pat = "|".join(escaped_tokens)

    merges = []
    vocab = {idx: bytes([idx]) for idx in range(256)}
    # vocab = gpt2_bytes_to_unicode()
    num_merges = vocab_size - 256
    
    procs = []
    results_q = mp.Queue()
    word_count, pair_stats, pair_indices = Counter(), Counter(), defaultdict(Counter)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            proc = mp.Process(target=pretokenize, args=(start, end, f, special_tok_pat, results_q))
            procs.append(proc)
            proc.start()
        
        for _ in range(len(procs)):
            wc = results_q.get()
            word_count.update(wc)
        
        for proc in procs: proc.join()

    sorted_word_count = list(word_count.items())
    pair_stats, pair_indices = get_pair_stats(sorted_word_count)

    for i in range(num_merges - len(special_tokens)):
        # top_pair = max(pair_stats, key=pair_stats.get)
        top_pair = max(pair_stats, key=lambda x: (pair_stats[x], x)) # chooses the lexicographically greater pair
        if pair_stats[top_pair] == 0:
            break
        vocab[len(vocab)] = top_pair[0] + top_pair[1]
        merges.append(top_pair)
        changes = replace_pair(top_pair, sorted_word_count, pair_indices)
        update_pair_stats(top_pair, changes, pair_stats, pair_indices)

    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")
    
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("../tests/fixtures/tinystories_sample_5M.txt", 400, ["<|endoftext|>"])
    # vocab, merges = train_bpe("../../data/TinyStoriesV2-GPT4-valid.txt", 400, ["<|endoftext|>"])
    print(vocab)
    print(merges)