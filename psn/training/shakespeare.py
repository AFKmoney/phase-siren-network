"""Shakespeare Data Pipeline for PSN training.

Character-level tokenization of Shakespeare's works. The dataset is
classic for language modeling benchmarks (Karpathy, 2015) and is ideal
for demonstrating the PSN because:

1. Rich character-level patterns (punctuation, capitalization, verse structure)
2. Small vocabulary (~67 unique characters) -- manageable for a prototype
3. Well-known baseline -- GPT-2 nano achieves ~1.1 bits/char on this
4. The rhythmic structure of Shakespeare's verse has natural phase-like properties

The data pipeline produces:
- token_ids: Integer sequences for network input
- target_ids: Next-token targets for adaptation
- Character-level encode/decode for text generation
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
import urllib.request
import os

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class ShakespeareData:
    """Character-level Shakespeare dataset.

    Handles downloading, tokenization, batching, and sequence generation.

    Attributes:
        text: Raw text string.
        char_to_idx: Mapping from character to integer ID.
        idx_to_char: Mapping from integer ID to character.
        vocab_size: Number of unique characters.
        sequences: List of (input_ids, target_ids) pairs.
    """

    SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(
        self,
        seq_len: int = 256,
        download_path: str = "/home/z/my-project/download/shakespeare.txt",
    ):
        self.seq_len = seq_len
        self.download_path = download_path
        self.text = ""
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        self.token_ids: np.ndarray = np.array([], dtype=np.int32)
        self.num_sequences = 0

        self._load_data()
        self._build_vocab()
        self._tokenize()

    def _load_data(self):
        """Download or load Shakespeare text."""
        if os.path.exists(self.download_path):
            with open(self.download_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
            print(f"[ShakespeareData] Loaded {len(self.text)} chars from {self.download_path}")
        else:
            print("[ShakespeareData] Downloading Shakespeare dataset...")
            try:
                urllib.request.urlretrieve(self.SHAKESPEARE_URL, self.download_path)
                with open(self.download_path, 'r', encoding='utf-8') as f:
                    self.text = f.read()
                print(f"[ShakespeareData] Downloaded {len(self.text)} chars")
            except Exception as e:
                print(f"[ShakespeareData] Download failed: {e}. Using built-in excerpt.")
                self.text = self._builtin_excerpt()

    def _builtin_excerpt(self) -> str:
        """Built-in Shakespeare excerpt if download fails."""
        return """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep—
No more—and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to—'tis a consummation
Devoutly to be wished. To die, to sleep—
To sleep—perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th' oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of th' unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveler returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry,
And lose the name of action.

Soft you now! The fair Ophelia! Nymph, in thy orisons
Be all my sins remembered.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honor, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side;
His youthful hose, well saved, a world too wide
For his shrunk shank, and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

If music be the food of love, play on,
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.
That strain again! it had a dying fall;
O, it came o'er my ear like the sweet sound
That breathes upon a bank of violets,
Stealing and giving odour! Enough, no more,
'Tis not so sweet now as it was before.
O spirit of love, how quick and fresh art thou,
That, notwithstanding thy capacity
Receiveth as the sea, nought enters there,
Of what validity and pitch soe'er,
But falls into abatement and low price
Even in a minute! So full of shapes is fancy
That it alone is high fantastical.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
Grim-visaged war hath smooth'd his wrinkled front;
And now, instead of mounting barbed steeds
To fright the souls of fearful adversaries,
He capers nimbly in a lady's chamber
To the lascivious pleasing of a lute.
But I, that am not shaped for sportive tricks,
Nor made to court an amorous looking-glass;
I, that am rudely stamp'd, and want love's majesty
To strut before a wanton ambling nymph;
I, that am curtail'd of this fair proportion,
Cheated of feature by dissembling nature,
Deform'd, unfinish'd, sent before my time
Into this breathing world, scarce half made up,
And that so lamely and unfashionable
That dogs bark at me as I halt by them;
Why, I, in this weak piping time of peace,
Have no delight to pass away the time,
Unless to spy my shadow in the sun
And descant on mine own deformity:
And therefore, since I cannot prove a lover,
To entertain these fair well-spoken days,
I am determined to prove a villain
And hate the idle pleasures of these days.
"""

    def _build_vocab(self):
        """Build character-to-index and index-to-character mappings."""
        unique_chars = sorted(set(self.text))
        self.char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        print(f"[ShakespeareData] Vocabulary size: {self.vocab_size}")
        print(f"[ShakespeareData] Characters: {''.join(unique_chars)}")

    def _tokenize(self):
        """Convert text to integer token IDs."""
        self.token_ids = np.array(
            [self.char_to_idx.get(ch, 0) for ch in self.text],
            dtype=np.int32,
        )
        # Number of complete sequences we can extract
        self.num_sequences = max(0, len(self.token_ids) - self.seq_len - 1)
        print(f"[ShakespeareData] Total tokens: {len(self.token_ids)}")
        print(f"[ShakespeareData] Usable sequences: {self.num_sequences}")

    def get_batch(self, batch_size: int, start_idx: int):
        """Get a batch of input/target sequences.

        Args:
            batch_size: Number of sequences in the batch.
            start_idx: Starting index in the token array.

        Returns:
            input_ids: (batch_size, seq_len) integer array.
            target_ids: (batch_size, seq_len) integer array (shifted by 1).
        """
        inputs = []
        targets = []
        for b in range(batch_size):
            idx = (start_idx + b * self.seq_len) % max(self.num_sequences, 1)
            inp = self.token_ids[idx: idx + self.seq_len]
            tgt = self.token_ids[idx + 1: idx + self.seq_len + 1]
            # Pad if needed
            if len(inp) < self.seq_len:
                inp = np.pad(inp, (0, self.seq_len - len(inp)), mode='constant')
            if len(tgt) < self.seq_len:
                tgt = np.pad(tgt, (0, self.seq_len - len(tgt)), mode='constant')
            inputs.append(inp)
            targets.append(tgt)

        if HAS_JAX:
            return (
                jnp.array(np.array(inputs), dtype=jnp.int32),
                jnp.array(np.array(targets), dtype=jnp.int32),
            )
        return (
            np.array(inputs, dtype=np.int32),
            np.array(targets, dtype=np.int32),
        )

    def encode(self, text: str) -> List[int]:
        """Encode a string to token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return "".join(
            self.idx_to_char.get(int(tid), "") for tid in token_ids
        )

    def get_dataset_stats(self) -> Dict[str, any]:
        """Get statistics about the dataset.

        Returns:
            Dictionary with character frequencies, sequence count, etc.
        """
        from collections import Counter
        char_counts = Counter(self.text)
        total = sum(char_counts.values())
        char_freqs = {ch: count / total for ch, count in char_counts.most_common()}

        return {
            "total_chars": len(self.text),
            "vocab_size": self.vocab_size,
            "unique_chars": list(self.char_to_idx.keys()),
            "char_frequencies": dict(list(char_freqs.items())[:20]),
            "num_sequences": self.num_sequences,
            "seq_len": self.seq_len,
        }
