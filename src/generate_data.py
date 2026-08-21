"""
generate_data.py
-----------------
Generates a synthetic dataset of short Nigerian Pidgin / English voice-or-text
commands, each labeled with the "app action" (intent) it maps to.

WHY SYNTHETIC DATA?
Real labeled Pidgin/English command datasets are scarce. To build a working
MVP quickly, we generate realistic command variations from templates that
mix formal English, Nigerian Pidgin, common typos, and casual phrasing -
the kind of input a low-literacy user would actually type or say.

HOW IT WORKS:
1. We define an "intent set" -> the list of app actions the classifier must
   recognize (e.g. check_balance, send_money, buy_airtime, etc.)
2. For each intent we define several *template sentences* with placeholders
   like {name}, {amount}, {app}, {bill}.
3. We fill in the placeholders with many different values (Nigerian names,
   amounts, common apps, etc.) to generate hundreds of unique commands.
4. We add light noise (extra filler words, punctuation, capitalization
   changes) to mimic real-world messy input.
5. Everything is shuffled and saved to data/commands_dataset.csv

Run this script directly to (re)generate the dataset:
    python src/generate_data.py
"""

import csv
import random
import os

# Fix the random seed so the dataset is reproducible every time we run this
# script (useful for grading / re-running the notebook and getting the same
# results).
random.seed(42)

# --------------------------------------------------------------------------
# 1. INTENT SET
# --------------------------------------------------------------------------
# These are the "app actions" our classifier must predict. This mirrors what
# a simple assistant app (banking / utility / phone-control app) would need
# to support for a low-literacy user base in Nigeria.
INTENTS = [
    "check_balance",     # "how much I get for account"
    "send_money",        # "send money give my mama"
    "buy_airtime",       # "buy credit / recharge card"
    "buy_data",          # "buy data bundle"
    "call_contact",      # "call my broda"
    "open_app",          # "open camera / open whatsapp"
    "check_weather",     # "how the weather be"
    "set_reminder",      # "remind me make I pay light bill"
    "play_music",        # "play music / play song"
    "stop_action",       # "stop am / cancel"
    "greeting",          # "how far / good morning"
    "help_request",      # "help me / wetin I go do"
]

# --------------------------------------------------------------------------
# 2. SLOT VALUES
# --------------------------------------------------------------------------
# Realistic Nigerian names, amounts, apps, contacts, and bills used to fill
# template placeholders so the generated sentences don't all look identical.
NAMES = [
    "mama", "papa", "my broda", "my sista", "Chidi", "Ngozi", "Tunde", "Amaka",
    "Emeka", "Blessing", "Ibrahim", "Kemi", "Uncle Femi", "Aunty Bisi", "my oga",
    "my guy", "Fatima", "Musa", "Chinedu", "Grace",
]

AMOUNTS = [
    "500 naira", "1000 naira", "2k", "5000 naira", "10k", "200 naira",
    "1500 naira", "3000 naira", "20k", "50 naira", "N500", "N2000",
]

APPS = [
    "camera", "whatsapp", "facebook", "gallery", "settings", "calculator",
    "phone book", "gmail", "youtube", "the browser", "instagram", "playstore",
]

BILLS = [
    "light bill", "school fees", "rent", "DSTV subscription", "water bill",
    "generator fuel", "shop rent", "electricity bill", "church offering",
]

TIMES = [
    "by 6am", "later today", "tomorrow morning", "by 5 o'clock", "this evening",
    "before 12", "next week", "on Monday", "by weekend",
]

NETWORKS = ["MTN", "Glo", "Airtel", "9mobile"]

SONG_OR_ARTIST = [
    "some music", "afrobeat", "Burna Boy", "gospel song", "Wizkid", "highlife",
    "my playlist", "fuji music", "Davido",
]

# --------------------------------------------------------------------------
# 3. TEMPLATES PER INTENT
# --------------------------------------------------------------------------
# Each intent has multiple template strings. "{slot}" placeholders get
# replaced with random values from the lists above. Mixing Pidgin and
# English templates ensures the model learns both registers.
TEMPLATES = {
    "check_balance": [
        "how much I get for my account",
        "check my balance",
        "abeg check my account balance",
        "wetin be my balance",
        "I wan know how much dey my account",
        "show me my account balance",
        "what is my current balance",
        "how much money I get now",
        "balance check abeg",
        "confirm my balance make I see",
    ],
    "send_money": [
        "send {amount} give {name}",
        "I wan send money to {name}",
        "transfer {amount} to {name}",
        "abeg send {amount} give {name} sharp sharp",
        "send money give {name} now now",
        "please transfer {amount} to {name}'s account",
        "I want to send {amount} to {name}",
        "make transfer of {amount} go {name}",
    ],
    "buy_airtime": [
        "buy {amount} airtime for me",
        "recharge my line with {amount}",
        "abeg buy credit give me",
        "top up my {network} line with {amount}",
        "I wan buy recharge card of {amount}",
        "load {amount} credit for my phone",
        "buy airtime {amount} on {network}",
    ],
    "buy_data": [
        "buy data for me",
        "I wan buy data bundle",
        "recharge my data with {amount}",
        "abeg buy {network} data bundle",
        "top up my data plan",
        "buy {amount} data on {network} for me",
        "load internet data for my phone",
    ],
    "call_contact": [
        "call {name}",
        "abeg call {name} for me",
        "dial {name}'s number",
        "phone {name} now",
        "I wan call {name}",
        "connect me to {name}",
        "ring {name}",
    ],
    "open_app": [
        "open {app}",
        "abeg open {app} for me",
        "launch {app}",
        "I wan use {app}",
        "start {app}",
        "bring up {app}",
        "show me {app}",
    ],
    "check_weather": [
        "how the weather be today",
        "wetin be the weather like",
        "is it going to rain today",
        "check weather for me",
        "abeg tell me the weather",
        "will rain fall today",
        "what is the weather forecast",
        "e go rain today?",
    ],
    "set_reminder": [
        "remind me make I pay {bill} {time}",
        "abeg remind me about {bill}",
        "set reminder for {bill} {time}",
        "I wan set alarm to remember {bill}",
        "don't make me forget {bill} {time}",
        "please remind me to pay {bill}",
        "set a reminder for me {time}",
    ],
    "play_music": [
        "play {song}",
        "abeg play {song} for me",
        "I wan hear {song}",
        "put on {song}",
        "start playing {song}",
        "play some music na",
        "gimme {song}",
    ],
    "stop_action": [
        "stop am",
        "cancel this one",
        "abeg stop",
        "pause everything",
        "no do am again",
        "stop the music",
        "cancel that action",
        "halt",
    ],
    "greeting": [
        "how far",
        "good morning",
        "how you dey",
        "good afternoon",
        "wetin dey happen",
        "hope you dey fine",
        "hello there",
        "good evening my friend",
    ],
    "help_request": [
        "help me abeg",
        "wetin I go do",
        "I no understand, help me",
        "abeg I need help",
        "how does this work",
        "I dey confuse, assist me",
        "please guide me",
        "I need assistance",
    ],
}

# --------------------------------------------------------------------------
# 4. NOISE FUNCTIONS
# --------------------------------------------------------------------------
# Real user input on a phone (typed fast, or transcribed from speech) is
# messy. We simulate this so the classifier is robust, not just memorizing
# clean templates.

FILLER_PREFIXES = ["", "", "", "abeg ", "please ", "biko ", "oga "]
FILLER_SUFFIXES = ["", "", "", " abeg", " please", " o", " na", " jare"]


def typo_noise(word):
    """
    Simulates a realistic typing/voice-transcription slip on a single word:
    - swap two adjacent letters, OR
    - drop one letter, OR
    - repeat one letter
    Only applied to words longer than 3 characters, so short function words
    (e.g. "am", "na", "I") stay intact and readable.
    """
    if len(word) <= 3:
        return word
    idx = random.randint(0, len(word) - 2)
    choice = random.random()
    if choice < 0.34:
        # swap two adjacent letters, e.g. "balance" -> "balnace"
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    elif choice < 0.67:
        # drop a letter, e.g. "balance" -> "balace"
        return word[:idx] + word[idx + 1:]
    else:
        # repeat a letter, e.g. "balance" -> "ballance"
        return word[:idx] + word[idx] + word[idx:]


def add_noise(sentence, typo_prob=0.12):
    """
    Randomly perturbs a sentence to mimic messy real-world text/voice input:
    - random filler words at the start/end
    - random capitalization of the first letter
    - occasional single-word typos (typo_prob chance PER WORD)
    - occasional missing punctuation / extra spaces

    typo_prob=0.12 means roughly 12% of eligible words in a sentence get a
    small spelling slip -- common with fast typing or noisy speech-to-text
    transcription, which this classifier must be robust to.
    """
    prefix = random.choice(FILLER_PREFIXES)
    suffix = random.choice(FILLER_SUFFIXES)
    sentence = f"{prefix}{sentence}{suffix}".strip()

    # Apply typos to a random subset of words
    words = sentence.split()
    words = [typo_noise(w) if random.random() < typo_prob else w for w in words]
    sentence = " ".join(words)

    # Randomly capitalize the first letter (simulates inconsistent typing)
    if random.random() < 0.5 and sentence:
        sentence = sentence[0].upper() + sentence[1:]

    # Occasionally collapse double spaces caused by empty prefix/suffix
    sentence = " ".join(sentence.split())
    return sentence


def fill_template(template: str) -> str:
    """Replaces {slot} placeholders in a template with a random slot value."""
    return template.format(
        name=random.choice(NAMES),
        amount=random.choice(AMOUNTS),
        app=random.choice(APPS),
        bill=random.choice(BILLS),
        time=random.choice(TIMES),
        network=random.choice(NETWORKS),
        song=random.choice(SONG_OR_ARTIST),
    )


# --------------------------------------------------------------------------
# 5. GENERATE THE FULL DATASET
# --------------------------------------------------------------------------
def generate_dataset(samples_per_intent: int = 80):
    """
    Builds a list of (command_text, intent_label) rows.

    For each intent, we repeatedly pick a random template, fill its slots,
    and apply noise until we reach `samples_per_intent` unique-ish examples.
    """
    rows = []
    for intent in INTENTS:
        templates = TEMPLATES[intent]
        seen = set()
        attempts = 0
        while len(seen) < samples_per_intent and attempts < samples_per_intent * 20:
            attempts += 1
            template = random.choice(templates)
            filled = fill_template(template)
            noisy = add_noise(filled)
            key = noisy.lower()
            if key not in seen:
                seen.add(key)
                rows.append((noisy, intent))
    random.shuffle(rows)
    return rows


def save_dataset(rows, path):
    """Writes the generated rows to a CSV file with header: command,intent"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["command", "intent"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    # Generate ~80 examples per intent (~960 rows total across 12 intents)
    dataset_rows = generate_dataset(samples_per_intent=80)
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "commands_dataset.csv"
    )
    save_dataset(dataset_rows, output_path)
