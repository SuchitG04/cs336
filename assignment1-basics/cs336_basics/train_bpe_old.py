import multiprocessing as mp
import os
import regex as re

from collections import Counter
from typing import BinaryIO

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


def get_word_count(text):
    word_count = Counter()
    for word in re.finditer(PAT, text, re.IGNORECASE):
        word = word.captures()[0]
        word_count[
            tuple([bytes([o]) for o in list(word.encode("utf-8"))])
        ] += 1
    return word_count


def get_pair_count(word_count):
    pair_count = Counter()
    for word, freq in word_count.items(): 
        for p in zip(word, word[1:]):
            pair_count[p] += freq

    return pair_count


def merge(
    word_count: dict[tuple[bytes], int],
    pair_count: dict[tuple[bytes, bytes], int],
    top_pair: tuple[bytes, bytes]
):
    new_word_count = Counter()
    for word, freq in word_count.items(): # word_count -> dict[tuple[bytes], int]
        new_word = ()
        i = 0
        idxs = []
        new_tok = top_pair[0] + top_pair[1]

        while i+1 < len(word):
            if (word[i], word[i+1]) == top_pair:
                idxs.append(i)
                new_word += (new_tok, )
                if i > 0:
                    pair_count[(word[i-1], word[i])] -= freq # decrement left neighbour
                if i+2 < len(word):
                    pair_count[(word[i+1], word[i+2])] -= freq # decrement right neighbour
                i += 2
            else:
                new_word += (word[i], )
                i += 1
        if i < len(word): new_word += (word[i], )

        # create and insert the new tuple
        new_word_count[new_word] = freq
        if idxs:
            # add new neighbours' counts
            for (p1, p2) in zip(new_word, new_word[1:]):
                if p1 == new_tok or p2 == new_tok:
                    pair_count[(p1, p2)] += freq

    return new_word_count


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
        wc = get_word_count(part)
        word_count.update(wc)
    pair_count = get_pair_count(word_count)
    
    proc_q.put((word_count, pair_count))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    escaped_tokens = [re.escape(o) for o in special_tokens]
    special_tok_pat = "|".join(escaped_tokens)

    merges = []
    vocab = {idx: bytes([idx]) for idx in range(256)}
    num_merges = vocab_size - 256
    
    procs = []
    results_q = mp.Queue()
    word_count = Counter()
    pair_count = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            proc = mp.Process(target=pretokenize, args=(start, end, f, special_tok_pat, results_q))
            procs.append(proc)
            proc.start()
        
        for _ in range(len(procs)):
            wc, pc = results_q.get()
            word_count.update(wc)
            pair_count.update(pc)
        
        for proc in procs: proc.join()


    for i in range(num_merges):
        new_byte_idx = 256 + i
        top_pair = max(pair_count, key=pair_count.get)
        if pair_count[top_pair] == 0:
            break
        vocab[new_byte_idx] = top_pair[0] + top_pair[1]
        del pair_count[top_pair]
        word_count = merge(word_count, pair_count, top_pair)
        merges.append(top_pair)

        mismatch_found = False
        for k_check, v_expected in get_pair_count(word_count).items():
            v_actual = pair_count.get(k_check, 0)
            if v_actual != v_expected:
                print(f"{i} MISMATCH for pair {k_check}: actual {v_actual}, expected {v_expected}")
                mismatch_found = True
        if mismatch_found:
            break
    
    return vocab, merges, word_count, pair_count

    