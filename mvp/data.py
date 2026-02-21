"""
Synthetic Training Data for Memory Graft MVP.

Generates fact-question-answer triples where the answer REQUIRES
information from the fact (which is stored in the memory bank).

The model cannot answer correctly from the question alone —
it MUST retrieve from memory.
"""

import random
import itertools

# === TEMPLATE POOL ===

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Elnur", "Fatima", "George",
    "Hannah", "Ivan", "Julia", "Karim", "Luna", "Marcus", "Nina",
    "Oscar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera",
    "Walter", "Xena", "Yuki", "Zara",
]

CITIES = [
    "London", "Tokyo", "Baku", "New York", "Berlin", "Paris", "Sydney",
    "Toronto", "Mumbai", "Cairo", "Seoul", "Istanbul", "Moscow",
    "San Francisco", "Amsterdam", "Barcelona", "Dubai", "Singapore",
    "Stockholm", "Vienna", "Lisbon", "Prague", "Bangkok", "Lima",
]

COLORS = [
    "blue", "red", "green", "purple", "orange", "black", "white",
    "yellow", "teal", "crimson", "silver", "gold",
]

FOODS = [
    "pizza", "sushi", "tacos", "pasta", "ramen", "curry", "burgers",
    "pho", "dumplings", "paella", "falafel", "pad thai", "biryani",
]

LANGUAGES = [
    "Python", "Rust", "JavaScript", "TypeScript", "Go", "Java", "C++",
    "Ruby", "Kotlin", "Swift", "Elixir", "Haskell",
]

DATABASES = [
    "PostgreSQL", "MongoDB", "Redis", "SQLite", "MySQL", "DynamoDB",
    "Cassandra", "Neo4j", "ClickHouse",
]

PETS = [
    ("dog", "Rex"), ("cat", "Whiskers"), ("dog", "Buddy"), ("cat", "Luna"),
    ("parrot", "Polly"), ("dog", "Max"), ("cat", "Milo"), ("hamster", "Nibbles"),
    ("dog", "Bella"), ("cat", "Oliver"), ("rabbit", "Flopsy"),
]

HOBBIES = [
    "painting", "chess", "hiking", "photography", "cooking", "gardening",
    "reading", "swimming", "cycling", "rock climbing", "piano", "yoga",
]

ALLERGIES = [
    "peanuts", "shellfish", "dairy", "gluten", "eggs", "soy",
    "tree nuts", "wheat", "sesame",
]

# === TEMPLATE DEFINITIONS ===
# Each template: (fact_template, question_variants, answer_template)

TEMPLATES = [
    # --- Identity ---
    {
        "fact": "The user's name is {name}.",
        "questions": [
            "What is my name?",
            "Do you remember my name?",
            "Who am I?",
            "Can you tell me my name?",
        ],
        "answer": "Your name is {name}.",
        "fill": lambda: {"name": random.choice(NAMES)},
    },
    # --- Location ---
    {
        "fact": "The user lives in {city}.",
        "questions": [
            "Where do I live?",
            "What city am I in?",
            "Do you know where I live?",
            "What's my city?",
        ],
        "answer": "You live in {city}.",
        "fill": lambda: {"city": random.choice(CITIES)},
    },
    # --- Preferences ---
    {
        "fact": "The user's favorite color is {color}.",
        "questions": [
            "What's my favorite color?",
            "Do you remember my favorite color?",
            "What color do I like?",
        ],
        "answer": "Your favorite color is {color}.",
        "fill": lambda: {"color": random.choice(COLORS)},
    },
    {
        "fact": "The user's favorite food is {food}.",
        "questions": [
            "What's my favorite food?",
            "What food do I like most?",
            "Do you remember what I like to eat?",
        ],
        "answer": "Your favorite food is {food}.",
        "fill": lambda: {"food": random.choice(FOODS)},
    },
    # --- Technical ---
    {
        "fact": "The user's preferred programming language is {lang}.",
        "questions": [
            "What programming language do I use?",
            "What's my preferred language?",
            "Which language do I like to code in?",
        ],
        "answer": "Your preferred programming language is {lang}.",
        "fill": lambda: {"lang": random.choice(LANGUAGES)},
    },
    {
        "fact": "The project uses {db} as its database.",
        "questions": [
            "What database are we using?",
            "Which database did we pick?",
            "What's our database?",
        ],
        "answer": "The project uses {db}.",
        "fill": lambda: {"db": random.choice(DATABASES)},
    },
    # --- Pets ---
    {
        "fact": "The user has a {pet_type} named {pet_name}.",
        "questions": [
            "Do I have any pets?",
            "What's my pet's name?",
            "Tell me about my pet.",
        ],
        "answer": "You have a {pet_type} named {pet_name}.",
        "fill": lambda: dict(zip(["pet_type", "pet_name"], random.choice(PETS))),
    },
    # --- Hobbies ---
    {
        "fact": "The user enjoys {hobby} in their free time.",
        "questions": [
            "What do I do for fun?",
            "What's my hobby?",
            "What do I enjoy doing?",
        ],
        "answer": "You enjoy {hobby}.",
        "fill": lambda: {"hobby": random.choice(HOBBIES)},
    },
    # --- Allergies (safety-relevant memory) ---
    {
        "fact": "The user is allergic to {allergy}.",
        "questions": [
            "Do I have any allergies?",
            "What am I allergic to?",
            "What should you avoid recommending to me?",
        ],
        "answer": "You are allergic to {allergy}.",
        "fill": lambda: {"allergy": random.choice(ALLERGIES)},
    },
    # --- Age ---
    {
        "fact": "The user is {age} years old.",
        "questions": [
            "How old am I?",
            "Do you know my age?",
            "What's my age?",
        ],
        "answer": "You are {age} years old.",
        "fill": lambda: {"age": random.randint(18, 70)},
    },
    # --- Occupation ---
    {
        "fact": "The user works as a {job}.",
        "questions": [
            "What do I do for work?",
            "What's my job?",
            "What's my occupation?",
        ],
        "answer": "You work as a {job}.",
        "fill": lambda: {
            "job": random.choice([
                "software engineer", "teacher", "doctor", "designer",
                "data scientist", "journalist", "architect", "musician",
                "lawyer", "nurse", "researcher", "chef", "pilot",
            ])
        },
    },
    # --- Timezone ---
    {
        "fact": "The user is in the {tz} timezone.",
        "questions": [
            "What timezone am I in?",
            "What's my timezone?",
            "Where am I time-wise?",
        ],
        "answer": "You are in the {tz} timezone.",
        "fill": lambda: {
            "tz": random.choice([
                "EST", "PST", "GMT", "CET", "JST", "IST", "AEST",
                "CST", "MST", "UTC+4", "UTC+3",
            ])
        },
    },
]


def generate_single():
    """Generate a single (fact, question, answer) triple."""
    template = random.choice(TEMPLATES)
    fill_values = template["fill"]()

    fact = template["fact"].format(**fill_values)
    question = random.choice(template["questions"]).format(**fill_values)
    answer = template["answer"].format(**fill_values)

    return {"fact": fact, "question": question, "answer": answer}


def generate_dataset(n=500, seed=42):
    """
    Generate n training examples.

    Returns list of dicts: [{"fact": str, "question": str, "answer": str}, ...]
    """
    random.seed(seed)
    dataset = []

    # Generate examples, ensuring variety
    for _ in range(n):
        dataset.append(generate_single())

    return dataset


def generate_multi_fact_dataset(n=200, n_distractors=4, seed=42):
    """
    Generate examples where the memory bank has multiple facts
    but only ONE is relevant to the question.

    This tests SELECTIVE retrieval.

    Returns list of dicts:
    [{"facts": [str, ...], "question": str, "answer": str, "target_idx": int}, ...]
    """
    random.seed(seed)
    dataset = []

    for _ in range(n):
        # Generate the target fact
        target = generate_single()

        # Generate distractor facts (different templates)
        distractors = []
        for _ in range(n_distractors):
            d = generate_single()
            # Make sure distractor is different enough
            while d["fact"] == target["fact"]:
                d = generate_single()
            distractors.append(d["fact"])

        # Shuffle facts, track which one is the target
        all_facts = distractors + [target["fact"]]
        random.shuffle(all_facts)
        target_idx = all_facts.index(target["fact"])

        dataset.append({
            "facts": all_facts,
            "question": target["question"],
            "answer": target["answer"],
            "target_idx": target_idx,
        })

    return dataset


def format_for_training(example, tokenizer, max_length=128):
    """
    Format a single example for training.

    Returns:
        input_ids: [max_length] — question + answer tokens
        labels: [max_length] — -100 for question tokens, real ids for answer tokens
        fact: str — the fact to encode into memory
    """
    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = encoded.input_ids.squeeze(0)  # [max_length]

    # Find where the answer starts (after "Answer: ")
    question_part = f"Question: {example['question']}\nAnswer: "
    question_tokens = tokenizer(question_part, add_special_tokens=False).input_ids
    answer_start = len(question_tokens)

    # Create labels: -100 for question, real ids for answer
    labels = input_ids.clone()
    labels[:answer_start] = -100

    # Also mask padding
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "fact": example["fact"],
    }


if __name__ == "__main__":
    # Quick test
    data = generate_dataset(n=10)
    for ex in data:
        print(f"FACT:     {ex['fact']}")
        print(f"QUESTION: {ex['question']}")
        print(f"ANSWER:   {ex['answer']}")
        print()

    multi = generate_multi_fact_dataset(n=3, n_distractors=3)
    for ex in multi:
        print(f"FACTS: {ex['facts']}")
        print(f"Q:     {ex['question']}")
        print(f"A:     {ex['answer']}")
        print(f"TARGET: fact #{ex['target_idx']}")
        print()
